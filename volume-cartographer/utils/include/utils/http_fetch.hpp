#pragma once

// S3/HTTP client for volume-cartographer.
//
// This is a thin C++ RAII shim over the vendored single-file C library
// `libs3` (libs/libs3). It keeps the exact public API the rest of VC was
// written against (utils::HttpClient / HttpResponse / AwsAuth /
// is_s3_url / parse_s3_url / s3_to_https) so call sites are unchanged,
// but all S3 transport, SigV4 signing, retry/backoff, and AWS
// credential resolution (env / AWS CLI / EC2 IMDSv2-cached / SSO / INI)
// now live in libs3 -- volume-cartographer no longer owns that code.
//
// libs3's only hard dependency is libcurl; curl does not leak into this
// header (it is a C API behind libs3.h).

#include <libs3.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// libs3 is always available (vendored), so the historical UTILS_HAS_CURL
// guard is now always true. Kept defined for the (few) translation units
// and CMake targets that test it.
#ifndef UTILS_HAS_CURL
#define UTILS_HAS_CURL 1
#endif

namespace utils {

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------
struct HttpResponse {
    long status_code = 0;
    std::vector<std::byte> body;
    std::string content_type;
    std::size_t content_length = 0;

    [[nodiscard]] bool ok() const noexcept { return status_code >= 200 && status_code < 300; }
    [[nodiscard]] bool not_found() const noexcept { return status_code == 404; }

    [[nodiscard]] std::string_view body_string() const noexcept {
        return {reinterpret_cast<const char*>(body.data()), body.size()};
    }
};

// ---------------------------------------------------------------------------
// HttpAuth -- non-S3 bearer/basic auth (kept for API compatibility)
// ---------------------------------------------------------------------------
struct HttpAuth {
    std::string bearer_token;
    std::string username;
    std::string password;

    [[nodiscard]] bool empty() const noexcept {
        return bearer_token.empty() && username.empty() && password.empty();
    }
};

// ---------------------------------------------------------------------------
// S3 URL utilities (delegate to libs3)
// ---------------------------------------------------------------------------
struct S3Url {
    std::string bucket;
    std::string key;
    std::string region; // empty if not specified
};

/// Recognises "s3://", "S3://", and "s3+REGION://".
[[nodiscard]] inline bool is_s3_url(std::string_view url) noexcept {
    return ::s3_url_is_s3(std::string{url}.c_str());
}

/// Parse an S3 URL. Supports "s3://bucket/key" and "s3+REGION://bucket/key".
[[nodiscard]] inline std::optional<S3Url> parse_s3_url(std::string_view url) {
    ::s3_url parsed{};
    if (::s3_url_parse(std::string{url}.c_str(), &parsed) != S3_OK)
        return std::nullopt;
    S3Url out{
        parsed.bucket ? parsed.bucket : "",
        parsed.key ? parsed.key : "",
        parsed.region ? parsed.region : "",
    };
    ::s3_url_free(&parsed);
    return out;
}

/// Convert a parsed S3 URL to its virtual-hosted HTTPS form.
[[nodiscard]] inline std::string s3_to_https(const S3Url& parsed) {
    ::s3_url u{};
    // s3_url_to_https only reads the fields; libs3 does not take ownership.
    u.bucket = const_cast<char*>(parsed.bucket.c_str());
    u.key = parsed.key.empty() ? nullptr : const_cast<char*>(parsed.key.c_str());
    u.region = parsed.region.empty() ? nullptr : const_cast<char*>(parsed.region.c_str());
    char buf[2048];
    if (::s3_url_to_https(&u, buf, sizeof(buf)) != S3_OK) {
        std::string r = "https://" + parsed.bucket;
        r += parsed.region.empty() ? ".s3.amazonaws.com"
                                   : ".s3." + parsed.region + ".amazonaws.com";
        if (!parsed.key.empty()) r += "/" + parsed.key;
        return r;
    }
    return std::string{buf};
}

/// Convert an S3 URL string to HTTPS (pass-through if not S3).
[[nodiscard]] inline std::string s3_to_https(std::string_view s3_url) {
    auto parsed = parse_s3_url(s3_url);
    if (!parsed)
        return std::string{s3_url};
    return s3_to_https(*parsed);
}

// ---------------------------------------------------------------------------
// AwsAuth -- AWS SigV4 credentials
// ---------------------------------------------------------------------------
struct AwsAuth {
    std::string access_key;
    std::string secret_key;
    std::string session_token; // optional (STS temporary credentials)
    std::string region;        // e.g. "us-east-1"

    [[nodiscard]] bool empty() const noexcept {
        return access_key.empty() && secret_key.empty();
    }

    /// Read credentials from environment variables only:
    ///   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    ///   AWS_SESSION_TOKEN, AWS_DEFAULT_REGION
    [[nodiscard]] static AwsAuth from_env() {
        AwsAuth a;
        ::s3_credentials c{};
        ::s3_credentials_from_env(&c); // empty fields if unset; copy regardless
        if (c.access_key)    a.access_key = c.access_key;
        if (c.secret_key)    a.secret_key = c.secret_key;
        if (c.session_token) a.session_token = c.session_token;
        if (c.region)        a.region = c.region;
        ::s3_credentials_free(&c);
        return a;
    }

    /// Full credential resolution via libs3. Tries, in order: explicit
    /// profile / $AWS_PROFILE -> EC2 IMDSv2 (queried directly, cached
    /// in-process, refreshed before STS expiry) -> SSO profiles in
    /// ~/.aws/config -> default `aws configure export-credentials` ->
    /// ~/.aws/credentials + config INI -> environment variables.
    /// Thread-safe and cache-served, so cheap to call per request.
    [[nodiscard]] static AwsAuth load(const std::string& profile = "default") {
        AwsAuth a;
        ::s3_credentials c{};
        const char* prof = (profile.empty() || profile == "default")
                               ? nullptr : profile.c_str();
        if (::s3_credentials_load(prof, &c) == S3_OK) {
            if (c.access_key)    a.access_key = c.access_key;
            if (c.secret_key)    a.secret_key = c.secret_key;
            if (c.session_token) a.session_token = c.session_token;
            if (c.region)        a.region = c.region;
        }
        ::s3_credentials_free(&c);
        return a;
    }
};

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------
class HttpClient final {
public:
    struct Config {
        HttpAuth auth{};
        AwsAuth aws_auth{};  // AWS SigV4 (takes precedence over auth if non-empty)
        // Optional: resolve AWS creds per-request instead of using the
        // static aws_auth above. Required for long-lived clients on EC2
        // instance-role (STS) credentials, which rotate every ~1-6h.
        std::function<AwsAuth()> aws_auth_provider{};
        std::chrono::seconds connect_timeout{10};
        std::chrono::seconds transfer_timeout{30};
        bool follow_redirects{true};
        std::size_t max_retries{3};
        std::string user_agent{"utils-http/1.0"};
    };

    HttpClient() : HttpClient(Config{}) {}

    explicit HttpClient(Config config)
    {
        state_ = std::make_shared<State>();
        state_->cfg = std::make_shared<Config>(std::move(config));
        const Config& c = *state_->cfg;

        ::s3_config scfg{};
        if (c.aws_auth_provider) {
            scfg.cred_provider = &State::cred_provider_thunk;
            scfg.cred_userdata = state_.get();
        } else if (!c.aws_auth.empty()) {
            state_->static_creds = c.aws_auth;
            fill_s3_creds(scfg.creds, state_->static_creds);
        } else if (!c.auth.bearer_token.empty()) {
            state_->bearer = c.auth.bearer_token;
            scfg.bearer_token = state_->bearer.c_str();
        } else if (!c.auth.username.empty()) {
            state_->basic_user = c.auth.username;
            state_->basic_pass = c.auth.password;
            scfg.basic_user = state_->basic_user.c_str();
            scfg.basic_pass = state_->basic_pass.c_str();
        }
        if (!c.aws_auth.region.empty()) {
            state_->region = c.aws_auth.region;
            scfg.region = state_->region.c_str();
        }
        scfg.connect_timeout_s = static_cast<long>(c.connect_timeout.count());
        scfg.transfer_timeout_s = static_cast<long>(c.transfer_timeout.count());
        scfg.max_retries = static_cast<int>(c.max_retries);
        scfg.follow_redirects = c.follow_redirects;
        state_->ua = c.user_agent;
        scfg.user_agent = state_->ua.c_str();

        state_->client.reset(::s3_client_new(&scfg));
    }

    // Flip the process-global abort flag: every in-flight libs3 transfer
    // returns promptly and pending retries bail. Use for fast shutdown so
    // a worker pool isn't stuck inside an S3 timeout.
    static void abortAll() noexcept { ::s3_global_abort(); }
    static void resetAbort() noexcept { ::s3_global_reset_abort(); }
    [[nodiscard]] static bool isAborted() noexcept { return ::s3_global_is_aborted(); }

    [[nodiscard]] HttpResponse get(std::string_view url) const {
        ::s3_response r{};
        ::s3_get(client(), c_str(url), &r);
        return convert(r);
    }

    [[nodiscard]] HttpResponse get_range(std::string_view url,
                                          std::size_t offset, std::size_t length) const {
        if (length == 0)
            return HttpResponse{};
        ::s3_response r{};
        ::s3_get_range(client(), c_str(url), offset, length, &r);
        return convert(r);
    }

    [[nodiscard]] HttpResponse head(std::string_view url) const {
        ::s3_response r{};
        ::s3_head(client(), c_str(url), &r);
        return convert(r);
    }

    [[nodiscard]] HttpResponse put(std::string_view url,
                                    std::span<const std::byte> data,
                                    std::string_view content_type = "application/octet-stream") const {
        ::s3_response r{};
        std::string ct{content_type};
        ::s3_put(client(), c_str(url), data.data(), data.size(),
                 ct.empty() ? nullptr : ct.c_str(), &r);
        return convert(r);
    }

    [[nodiscard]] HttpResponse put_file(std::string_view url,
                                         const std::filesystem::path& file_path,
                                         std::string_view content_type = "application/octet-stream") const {
        ::s3_response r{};
        std::string ct{content_type};
        ::s3_put_file(client(), c_str(url), file_path.c_str(),
                      ct.empty() ? nullptr : ct.c_str(), &r);
        return convert(r);
    }

    [[nodiscard]] HttpResponse del(std::string_view url) const {
        ::s3_response r{};
        ::s3_delete(client(), c_str(url), &r);
        return convert(r);
    }

    // ListObjectsV2 with auto-pagination. `s3_url_prefix` is an s3:// URL
    // whose key part is the listing prefix; `delimiter` is "/" for one
    // level or nullptr for a fully recursive listing. `cb` is invoked once
    // per page (return true to continue paginating). Returns S3_OK on
    // success. libs3 owns SigV4 query signing, pagination, and XML parsing.
    [[nodiscard]] ::s3_status list_all(std::string_view s3_url_prefix,
                                       const char* delimiter,
                                       ::s3_list_page_fn cb,
                                       void* userdata) const {
        return ::s3_list_all(client(), c_str(s3_url_prefix), delimiter, cb, userdata);
    }

private:
    struct ClientDeleter {
        void operator()(::s3_client* c) const noexcept { ::s3_client_free(c); }
    };

    // Owned backing storage for everything s3_config / cred_provider points
    // at; shared so HttpClient stays copyable like the old value type was.
    struct State {
        std::shared_ptr<Config> cfg;
        std::unique_ptr<::s3_client, ClientDeleter> client;
        AwsAuth static_creds;
        std::string region, bearer, basic_user, basic_pass, ua;

        static ::s3_status cred_provider_thunk(void* ud, ::s3_credentials* out) {
            auto* self = static_cast<State*>(ud);
            AwsAuth a = self->cfg->aws_auth_provider
                            ? self->cfg->aws_auth_provider()
                            : self->cfg->aws_auth;
            if (a.region.empty()) a.region = self->cfg->aws_auth.region;
            if (a.empty()) return S3_ERR_NO_CREDS;
            out->access_key    = dup(a.access_key);
            out->secret_key    = dup(a.secret_key);
            out->session_token = a.session_token.empty() ? nullptr : dup(a.session_token);
            out->region        = a.region.empty() ? nullptr : dup(a.region);
            return S3_OK;
        }

        static char* dup(const std::string& s) {
            char* p = static_cast<char*>(std::malloc(s.size() + 1));
            if (p) std::memcpy(p, s.c_str(), s.size() + 1);
            return p;
        }
    };

    static void fill_s3_creds(::s3_credentials& c, AwsAuth& a) {
        c.access_key    = a.access_key.empty() ? nullptr : a.access_key.data();
        c.secret_key    = a.secret_key.empty() ? nullptr : a.secret_key.data();
        c.session_token = a.session_token.empty() ? nullptr : a.session_token.data();
        c.region        = a.region.empty() ? nullptr : a.region.data();
    }

    [[nodiscard]] ::s3_client* client() const { return state_->client.get(); }

    // libs3 wants NUL-terminated; cache the conversion per thread.
    static const char* c_str(std::string_view sv) {
        thread_local std::string buf;
        buf.assign(sv);
        return buf.c_str();
    }

    static HttpResponse convert(::s3_response& r) {
        HttpResponse out;
        out.status_code = r.status;
        if (r.body && r.body_len) {
            const auto* p = reinterpret_cast<const std::byte*>(r.body);
            out.body.assign(p, p + r.body_len);
        }
        if (r.content_type) out.content_type = r.content_type;
        out.content_length = static_cast<std::size_t>(r.content_length);
        ::s3_response_free(&r);
        return out;
    }

    std::shared_ptr<State> state_;
};

} // namespace utils

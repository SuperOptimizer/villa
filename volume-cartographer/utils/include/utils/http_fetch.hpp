#pragma once

// Thin C++ RAII shim over the vendored libs3 C library (libs/libs3).
// Keeps the public API VC was written against (HttpClient / HttpResponse
// / AwsAuth / parse_s3_url / s3_to_https); libs3 owns S3 transport,
// SigV4, retry, and AWS credential resolution.

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

// libs3 is always vendored; kept for call sites that test this.
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
// S3 listing (one page)
// ---------------------------------------------------------------------------
struct S3Object {
    std::string key;
    std::uint64_t size = 0;
    std::string etag; // may be empty
};

/// One page of a ListObjectsV2 response. `prefixes` are the CommonPrefixes
/// (sub-"directories") when a delimiter was used; `objects` are the keys.
/// When `is_truncated`, pass `next_continuation_token` back into list() for
/// the next page.
struct S3ListPage {
    std::vector<std::string> prefixes;
    std::vector<S3Object> objects;
    std::string next_continuation_token; // empty when not truncated
    bool is_truncated = false;
    ::s3_status status = S3_OK;
};

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------
class HttpClient final {
public:
    struct Config {
        HttpAuth auth{};
        AwsAuth aws_auth{};  // AWS SigV4 (takes precedence over auth if non-empty)
        // Per-request cred resolution; required for long-lived clients on
        // rotating EC2 instance-role (STS) credentials.
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

    // Process-global abort: in-flight transfers return promptly and
    // pending retries bail. For fast shutdown.
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

    // ListObjectsV2, auto-paginated. delimiter "/" = one level, nullptr =
    // recursive. cb is called per page; return true to keep paginating.
    [[nodiscard]] ::s3_status list_all(std::string_view s3_url_prefix,
                                       const char* delimiter,
                                       ::s3_list_page_fn cb,
                                       void* userdata) const {
        return ::s3_list_all(client(), c_str(s3_url_prefix), delimiter, cb, userdata);
    }

    // Single-page ListObjectsV2, for incremental UI browsing. delimiter
    // "/" returns CommonPrefixes; pass a prior next_continuation_token to
    // page forward.
    [[nodiscard]] S3ListPage list(std::string_view s3_url_prefix,
                                  const char* delimiter,
                                  std::string_view continuation_token = {}) const {
        S3ListPage out;
        ::s3_list_params params{};
        params.delimiter = delimiter;
        std::string tok{continuation_token};
        params.continuation_token = tok.empty() ? nullptr : tok.c_str();
        ::s3_list_result r{};
        out.status = ::s3_list_ex(client(), c_str(s3_url_prefix), &params, &r);
        if (out.status == S3_OK) {
            out.prefixes.reserve(r.prefix_count);
            for (size_t i = 0; i < r.prefix_count; ++i)
                out.prefixes.emplace_back(r.prefixes[i] ? r.prefixes[i] : "");
            out.objects.reserve(r.object_count);
            for (size_t i = 0; i < r.object_count; ++i)
                out.objects.push_back(S3Object{
                    r.objects[i].key ? r.objects[i].key : "",
                    r.objects[i].size,
                    r.objects[i].etag ? r.objects[i].etag : ""});
            if (r.next_continuation_token)
                out.next_continuation_token = r.next_continuation_token;
            out.is_truncated = r.is_truncated;
        }
        ::s3_list_result_free(&r);
        return out;
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

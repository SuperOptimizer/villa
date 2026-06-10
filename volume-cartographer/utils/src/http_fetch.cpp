// utils::HttpClient — thin adapter over libs3 (libs/libs3). libs3 owns the curl
// handles (one per thread per client), SigV4 signing, credential resolution,
// retries, and the process-wide abort flag; this file just maps the C++ API.
#if __has_include(<curl/curl.h>)

#include <utils/http_fetch.hpp>

#include "libs3.h"

#include <cstring>
#include <utility>

namespace utils {

namespace {

HttpResponse toResponse(s3_response& r)
{
    HttpResponse out;
    out.status_code = r.status;
    if (r.body && r.body_len) {
        const auto* p = reinterpret_cast<const std::byte*>(r.body);
        out.body.assign(p, p + r.body_len);
    }
    if (r.content_type)
        out.content_type = r.content_type;
    out.content_length = static_cast<std::size_t>(r.content_length);
    s3_response_free(&r);
    return out;
}

std::shared_ptr<void> makeClient(const HttpClient::Config& config)
{
    s3_config cfg{};
    // string lifetimes only need to span s3_client_new (it copies everything).
    if (!config.aws_auth.empty()) {
        cfg.creds.access_key = const_cast<char*>(config.aws_auth.access_key.c_str());
        cfg.creds.secret_key = const_cast<char*>(config.aws_auth.secret_key.c_str());
        cfg.creds.session_token = config.aws_auth.session_token.empty()
            ? nullptr : const_cast<char*>(config.aws_auth.session_token.c_str());
        cfg.creds.region = config.aws_auth.region.empty()
            ? nullptr : const_cast<char*>(config.aws_auth.region.c_str());
    } else if (!config.auth.empty()) {
        if (!config.auth.bearer_token.empty())
            cfg.bearer_token = config.auth.bearer_token.c_str();
        if (!config.auth.username.empty()) {
            cfg.basic_user = config.auth.username.c_str();
            cfg.basic_pass = config.auth.password.c_str();
        }
    }
    cfg.connect_timeout_s = config.connect_timeout.count();
    cfg.transfer_timeout_s = config.transfer_timeout.count();
    cfg.max_retries = static_cast<int>(config.max_retries);
    cfg.follow_redirects = config.follow_redirects;
    cfg.user_agent = config.user_agent.c_str();

    return {s3_client_new(&cfg), [](void* c) { s3_client_free(static_cast<s3_client*>(c)); }};
}

s3_client* client(const std::shared_ptr<void>& p)
{
    return static_cast<s3_client*>(p.get());
}

} // namespace

// ---------------------------------------------------------------------------
// S3 URL utilities
// ---------------------------------------------------------------------------
bool is_s3_url(std::string_view url) noexcept
{
    return s3_url_is_s3(std::string(url).c_str());
}

std::optional<S3Url> parse_s3_url(std::string_view url)
{
    s3_url u{};
    if (s3_url_parse(std::string(url).c_str(), &u) != S3_OK)
        return std::nullopt;
    S3Url out{u.bucket ? u.bucket : "", u.key ? u.key : "", u.region ? u.region : ""};
    s3_url_free(&u);
    return out;
}

std::string s3_to_https(const S3Url& parsed)
{
    s3_url u{};
    u.bucket = const_cast<char*>(parsed.bucket.c_str());
    u.key = const_cast<char*>(parsed.key.c_str());
    u.region = const_cast<char*>(parsed.region.c_str());
    char buf[2048];
    if (s3_url_to_https(&u, buf, sizeof buf) != S3_OK)
        return {};
    return buf;
}

std::string s3_to_https(std::string_view url)
{
    auto parsed = parse_s3_url(url);
    return parsed ? s3_to_https(*parsed) : std::string(url);
}

// ---------------------------------------------------------------------------
// AwsAuth
// ---------------------------------------------------------------------------
AwsAuth AwsAuth::load(const std::string& profile)
{
    s3_credentials c{};
    AwsAuth out;
    if (s3_credentials_load(profile.empty() ? nullptr : profile.c_str(), &c) == S3_OK) {
        if (c.access_key)    out.access_key = c.access_key;
        if (c.secret_key)    out.secret_key = c.secret_key;
        if (c.session_token) out.session_token = c.session_token;
        if (c.region)        out.region = c.region;
    }
    s3_credentials_free(&c);
    return out;
}

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------
HttpClient::HttpClient() : client_(makeClient(config_)) {}

HttpClient::HttpClient(Config config)
    : config_(std::move(config))
    , client_(makeClient(config_))
{
}

void HttpClient::abortAll() noexcept { s3_global_abort(); }
void HttpClient::resetAbort() noexcept { s3_global_reset_abort(); }
bool HttpClient::isAborted() noexcept { return s3_global_is_aborted(); }

HttpResponse HttpClient::get(std::string_view url) const
{
    s3_response r{};
    s3_get(client(client_), std::string(url).c_str(), &r);
    return toResponse(r);
}

HttpResponse HttpClient::get_range(std::string_view url,
                                   std::size_t offset, std::size_t length) const
{
    s3_response r{};
    s3_get_range(client(client_), std::string(url).c_str(), offset, length, &r);
    return toResponse(r);
}

HttpResponse HttpClient::head(std::string_view url) const
{
    s3_response r{};
    s3_head(client(client_), std::string(url).c_str(), &r);
    return toResponse(r);
}

HttpResponse HttpClient::put(std::string_view url,
                             std::span<const std::byte> data,
                             std::string_view content_type) const
{
    s3_response r{};
    s3_put(client(client_), std::string(url).c_str(), data.data(), data.size(),
           std::string(content_type).c_str(), &r);
    return toResponse(r);
}

HttpResponse HttpClient::put_file(std::string_view url,
                                  const std::filesystem::path& file_path,
                                  std::string_view content_type) const
{
    s3_response r{};
    s3_put_file(client(client_), std::string(url).c_str(), file_path.string().c_str(),
                std::string(content_type).c_str(), &r);
    return toResponse(r);
}

} // namespace utils

#endif // __has_include(<curl/curl.h>)

#include "vc/core/util/RemoteUrl.hpp"

#include <cstdlib>

namespace vc {

ResolvedUrl resolveRemoteUrl(const std::string& input)
{
    // Check for s3:// or s3+REGION:// prefix
    if (input.rfind("s3://", 0) == 0) {
        // s3://bucket/key — default region us-east-1
        std::string rest = input.substr(5);  // after "s3://"
        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string region = "us-east-1";
        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        bool hasCreds = std::getenv("AWS_ACCESS_KEY_ID") != nullptr;

        return ResolvedUrl{httpsUrl, region, hasCreds};
    }

    if (input.rfind("s3+", 0) == 0) {
        // s3+REGION://bucket/key
        auto schemeEnd = input.find("://");
        if (schemeEnd == std::string::npos) {
            // Malformed — treat as plain URL
            return ResolvedUrl{input, {}, false};
        }

        std::string region = input.substr(3, schemeEnd - 3);  // between "s3+" and "://"
        std::string rest = input.substr(schemeEnd + 3);

        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        bool hasCreds = std::getenv("AWS_ACCESS_KEY_ID") != nullptr;

        return ResolvedUrl{httpsUrl, region, hasCreds};
    }

    // HTTP/HTTPS — pass through
    return ResolvedUrl{input, {}, false};
}

}  // namespace vc

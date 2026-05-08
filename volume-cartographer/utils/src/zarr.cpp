#include "utils/zarr.hpp"

namespace utils {

// ---------------------------------------------------------------------------
// JSON wrapper functions
// ---------------------------------------------------------------------------

JsonValue json_parse(std::string_view text) {
    return Json::parse(text);
}

std::string json_serialize(const JsonValue& v, int indent) {
    return v.dump(indent);
}

const JsonValue* json_find(const JsonValue& obj, const std::string& key) {
    if (!obj.is_object()) return nullptr;
    if (!obj.contains(key)) return nullptr;
    return &obj.at(key);
}

// ---------------------------------------------------------------------------
// detail namespace
// ---------------------------------------------------------------------------

namespace detail {

ZarrMetadata parse_zarray(std::string_view json_str) {
    auto root = json_parse(json_str);
    if (!root.is_object())
        throw std::runtime_error("zarr: .zarray root must be a JSON object");

    ZarrMetadata meta;
    meta.version = ZarrVersion::v2;

    // shape
    if (auto* p = json_find(root, "shape"); p && p->is_array())
        for (const auto& v : (*p))
            meta.shape.push_back(v.get_size_t());

    // chunks
    if (auto* p = json_find(root, "chunks"); p && p->is_array())
        for (const auto& v : (*p))
            meta.chunks.push_back(v.get_size_t());

    // dtype (e.g. "<u2")
    if (auto* p = json_find(root, "dtype"); p && p->is_string()) {
        const auto& ds = p->get_string();
        if (!ds.empty() && (ds[0] == '<' || ds[0] == '>' || ds[0] == '|'))
            meta.byte_order = ds[0];
        auto dt = parse_dtype(ds);
        if (!dt) throw std::runtime_error("zarr: unsupported dtype: " + std::string(ds));
        meta.dtype = *dt;
    }

    // compressor
    if (auto* p = json_find(root, "compressor"); p) {
        if (p->is_object()) {
            if (auto* cid = json_find(*p, "id"); cid && cid->is_string())
                meta.compressor_id = cid->get_string();
            if (auto* cl = json_find(*p, "clevel"); cl && cl->is_number())
                meta.compression_level = cl->get_int();
        }
    }

    // fill_value
    if (auto* p = json_find(root, "fill_value"); p) {
        if (p->is_number())
            meta.fill_value = p->get_double();
        else if (p->is_null())
            meta.fill_value = std::nullopt;
    }

    // dimension_separator
    if (auto* p = json_find(root, "dimension_separator"); p && p->is_string())
        meta.dimension_separator = p->get_string();

    // filters: not parsed (delta, fixedscaleoffset, quantize are unused)

    return meta;
}

ZarrCodecConfig parse_codec_config(const JsonValue& jv) {
    ZarrCodecConfig cc;
    if (auto* n = json_find(jv, "name"); n && n->is_string())
        cc.name = n->get_string();
    if (auto* c = json_find(jv, "configuration"); c)
        cc.configuration = std::make_shared<JsonValue>(*c);
    return cc;
}

ZarrMetadata parse_zarr_json(std::string_view json_str) {
    auto root = json_parse(json_str);
    if (!root.is_object())
        throw std::runtime_error("zarr: zarr.json root must be a JSON object");

    ZarrMetadata meta;
    meta.version = ZarrVersion::v3;

    // node_type
    if (auto* p = json_find(root, "node_type"); p && p->is_string())
        meta.node_type = p->get_string();

    // shape
    if (auto* p = json_find(root, "shape"); p && p->is_array())
        for (const auto& v : (*p))
            meta.shape.push_back(v.get_size_t());

    // data_type
    if (auto* p = json_find(root, "data_type"); p && p->is_string()) {
        auto dt = parse_dtype_v3(p->get_string());
        if (!dt) throw std::runtime_error("zarr: unsupported v3 data_type: " + p->get_string());
        meta.dtype = *dt;
    }

    // chunk_grid
    if (auto* p = json_find(root, "chunk_grid"); p && p->is_object()) {
        if (auto* cfg = json_find(*p, "configuration"); cfg && cfg->is_object()) {
            if (auto* cs = json_find(*cfg, "chunk_shape"); cs && cs->is_array())
                for (const auto& v : (*cs))
                    meta.chunks.push_back(v.get_size_t());
        }
    }

    // chunk_key_encoding
    if (auto* p = json_find(root, "chunk_key_encoding"); p && p->is_object()) {
        if (auto* nm = json_find(*p, "name"); nm && nm->is_string())
            meta.chunk_key_encoding = nm->get_string();
        if (auto* cfg = json_find(*p, "configuration"); cfg && cfg->is_object()) {
            if (auto* sep = json_find(*cfg, "separator"); sep && sep->is_string())
                meta.dimension_separator = sep->get_string();
        }
    }
    // Default separators.
    if (meta.chunk_key_encoding == "default" && meta.dimension_separator == ".")
        meta.dimension_separator = "/";

    // fill_value
    if (auto* p = json_find(root, "fill_value"); p) {
        if (p->is_number())
            meta.fill_value = p->get_double();
        else if (p->is_null())
            meta.fill_value = std::nullopt;
    }

    // codecs
    if (auto* p = json_find(root, "codecs"); p && p->is_array()) {
        for (const auto& cv : (*p)) {
            if (!cv.is_object()) continue;
            auto cc = parse_codec_config(cv);

            // Detect sharding_indexed codec.
            if (cc.name == "sharding_indexed" && cc.configuration && cc.configuration->is_object()) {
                ShardConfig sc;
                const auto& cfg = *cc.configuration;
                if (auto* cs = json_find(cfg, "chunk_shape"); cs && cs->is_array())
                    for (const auto& v : (*cs))
                        sc.sub_chunks.push_back(v.get_size_t());
                // index_location is always "start" — ignore any "end" values
                if (auto* ic = json_find(cfg, "index_codecs"); ic && ic->is_array())
                    for (const auto& icv : (*ic))
                        if (icv.is_object()) sc.index_codecs.push_back(parse_codec_config(icv));
                if (auto* sc_codecs = json_find(cfg, "codecs"); sc_codecs && sc_codecs->is_array())
                    for (const auto& scv : (*sc_codecs))
                        if (scv.is_object()) sc.sub_codecs.push_back(parse_codec_config(scv));
                meta.shard_config = std::move(sc);
            }

            meta.codecs.push_back(std::move(cc));
        }
    }

    return meta;
}

JsonValue codec_config_to_json(const ZarrCodecConfig& cc) {
    JsonObject obj;
    obj["name"] = JsonValue(cc.name);
    if (cc.configuration && !cc.configuration->is_null())
        obj["configuration"] = *cc.configuration;
    return JsonValue(std::move(obj));
}

std::string serialize_zarr_json(const ZarrMetadata& meta) {
    JsonObject root;
    root["zarr_format"] = JsonValue(3);
    root["node_type"] = JsonValue(meta.node_type);

    // data_type
    root["data_type"] = JsonValue(std::string(dtype_string_v3(meta.dtype)));

    // shape
    {
        JsonArray arr;
        for (auto s : meta.shape) arr.push_back(JsonValue(s));
        root["shape"] = JsonValue(std::move(arr));
    }

    // chunk_grid
    {
        JsonObject cg;
        cg["name"] = JsonValue("regular");
        JsonObject cg_cfg;
        JsonArray cs;
        for (auto c : meta.chunks) cs.push_back(JsonValue(c));
        cg_cfg["chunk_shape"] = JsonValue(std::move(cs));
        cg["configuration"] = JsonValue(std::move(cg_cfg));
        root["chunk_grid"] = JsonValue(std::move(cg));
    }

    // chunk_key_encoding
    {
        JsonObject cke;
        cke["name"] = JsonValue(meta.chunk_key_encoding);
        if (meta.chunk_key_encoding == "v2") {
            JsonObject cke_cfg;
            cke_cfg["separator"] = JsonValue(meta.dimension_separator);
            cke["configuration"] = JsonValue(std::move(cke_cfg));
        }
        root["chunk_key_encoding"] = JsonValue(std::move(cke));
    }

    // fill_value
    if (meta.fill_value.has_value())
        root["fill_value"] = JsonValue(*meta.fill_value);
    else
        root["fill_value"] = JsonValue(nullptr);

    // codecs
    {
        JsonArray codecs_arr;

        // If we have a shard_config but no codecs list was provided, build one.
        if (meta.codecs.empty() && meta.shard_config) {
            // Build sharding_indexed codec entry.
            JsonObject sc_cfg;
            {
                JsonArray sub_cs;
                for (auto c : meta.shard_config->sub_chunks)
                    sub_cs.push_back(JsonValue(c));
                sc_cfg["chunk_shape"] = JsonValue(std::move(sub_cs));
            }
            sc_cfg["index_location"] = JsonValue("start");

            {
                JsonArray idx_codecs;
                for (const auto& ic : meta.shard_config->index_codecs)
                    idx_codecs.push_back(codec_config_to_json(ic));
                if (idx_codecs.empty()) {
                    // Default: bytes codec for index.
                    JsonObject bytes_codec;
                    bytes_codec["name"] = JsonValue("bytes");
                    JsonObject bytes_cfg;
                    bytes_cfg["endian"] = JsonValue("little");
                    bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
                    idx_codecs.push_back(JsonValue(std::move(bytes_codec)));
                }
                sc_cfg["index_codecs"] = JsonValue(std::move(idx_codecs));
            }

            {
                JsonArray sub_codecs;
                for (const auto& sc : meta.shard_config->sub_codecs)
                    sub_codecs.push_back(codec_config_to_json(sc));
                if (sub_codecs.empty()) {
                    JsonObject bytes_codec;
                    bytes_codec["name"] = JsonValue("bytes");
                    JsonObject bytes_cfg;
                    bytes_cfg["endian"] = JsonValue("little");
                    bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
                    sub_codecs.push_back(JsonValue(std::move(bytes_codec)));
                }
                sc_cfg["codecs"] = JsonValue(std::move(sub_codecs));
            }

            JsonObject sharding_obj;
            sharding_obj["name"] = JsonValue("sharding_indexed");
            sharding_obj["configuration"] = JsonValue(std::move(sc_cfg));
            codecs_arr.push_back(JsonValue(std::move(sharding_obj)));
        } else if (meta.codecs.empty()) {
            // Default: bytes codec.
            JsonObject bytes_codec;
            bytes_codec["name"] = JsonValue("bytes");
            JsonObject bytes_cfg;
            bytes_cfg["endian"] = JsonValue("little");
            bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
            codecs_arr.push_back(JsonValue(std::move(bytes_codec)));
        } else {
            for (const auto& cc : meta.codecs)
                codecs_arr.push_back(codec_config_to_json(cc));
        }

        root["codecs"] = JsonValue(std::move(codecs_arr));
    }

    return json_serialize(JsonValue(std::move(root)), 2) + "\n";
}

} // namespace detail

bool ZarrArray::needs_byteswap() const noexcept {
    auto elem_sz = dtype_size(meta_.dtype);
    if (elem_sz <= 1) return false;
    bool native_le = detail::is_little_endian();
    if (meta_.version == ZarrVersion::v2) {
        if (meta_.byte_order == '|') return false;
        return (meta_.byte_order == '<') != native_le;
    }
    for (const auto& cc : meta_.codecs) {
        if (cc.name == "bytes" && cc.configuration && cc.configuration->is_object()) {
            auto* e = json_find(*cc.configuration, "endian");
            if (e && e->is_string()) {
                bool stored_le = (e->get_string() == "little");
                return stored_le != native_le;
            }
        }
    }
    return false;
}

std::optional<std::vector<std::byte>>
ZarrArray::extract_inner_chunk(std::span<const std::byte> shard_data,
                               std::span<const std::size_t> inner_indices) const {
    const auto& sc = *meta_.shard_config;
    const auto n_inner = meta_.total_sub_chunks_per_shard();
    const std::size_t index_size = n_inner * 16;

    if (shard_data.size() < index_size)
        throw std::runtime_error("zarr: shard too small to contain index");

    // Index is always at the start of the shard.
    std::span<const std::byte> index_data = shard_data.subspan(0, index_size);

    std::vector<std::byte> decoded_index;
    if (!sc.index_codecs.empty()) {
        decoded_index.assign(index_data.begin(), index_data.end());
        for (const auto& ic : sc.index_codecs) {
            if (ic.name == "bytes") {
                if (ic.configuration && ic.configuration->is_object()) {
                    auto* e = json_find(*ic.configuration, "endian");
                    if (e && e->is_string() &&
                        e->get_string() == "big" &&
                        detail::is_little_endian()) {
                        detail::byteswap_inplace(decoded_index, 8);
                    } else if (e && e->is_string() &&
                               e->get_string() == "little" &&
                               !detail::is_little_endian()) {
                        detail::byteswap_inplace(decoded_index, 8);
                    }
                }
            } else if (codec_.decompress) {
                auto it = registry_.find(ic.name);
                if (it != registry_.end() && it->second.decompress) {
                    decoded_index = it->second.decompress(
                        decoded_index, n_inner * 16);
                }
            }
        }
        index_data = decoded_index;
    }

    auto index = detail::ShardIndex::deserialize(index_data, n_inner);

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = inner_indices.size(); d-- > 0;) {
        linear += inner_indices[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    if (linear >= n_inner)
        throw std::runtime_error("zarr: inner chunk index out of range");

    const auto& entry = index.entries[linear];
    if (entry.is_missing()) return std::nullopt;

    if (entry.offset + entry.nbytes > shard_data.size())
        throw std::runtime_error("zarr: inner chunk offset/size exceeds shard data");

    std::vector<std::byte> chunk(
        shard_data.begin() + static_cast<std::ptrdiff_t>(entry.offset),
        shard_data.begin() + static_cast<std::ptrdiff_t>(entry.offset + entry.nbytes));

    if (codec_.decompress && needs_decompression()) {
        chunk = codec_.decompress(chunk, meta_.sub_chunk_byte_size());
    }

    if (needs_byteswap()) {
        detail::byteswap_inplace(chunk, dtype_size(meta_.dtype));
    }

    return chunk;
}


} // namespace utils

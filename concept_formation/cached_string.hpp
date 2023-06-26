#ifndef CACHED_STRING_HPP
#define CACHED_STRING_HPP

#include <string>
#include <functional>
#include <unordered_map>

class CachedString {
public:
    CachedString(const std::string& s) : str(s),
          hash_value(std::hash<std::string>{}(s)),
          hidden(s.size() > 0 && s[0] == '_') {}

    CachedString(const char* s)
        : CachedString(std::string(s)) {}

    std::string get_string() const {
        return str;
    }

    size_t get_hash() const {
        return hash_value;
    }

    bool is_hidden() const {
        return hidden;
    }

    bool operator==(const CachedString& other) const {
        return str == other.str;
    }

    bool operator!=(const CachedString& other) const {
        return !(*this == other);
    }

    bool operator<(const CachedString& other) const {
        return str < other.str;
    }

private:
    std::string str;
    size_t hash_value;
    bool hidden;
};

namespace std {
    template <>
    struct hash<CachedString> {
        std::size_t operator()(const CachedString& cs) const {
            return cs.get_hash();
        }
    };

    template <>
    struct equal_to<CachedString> {
        bool operator()(const CachedString& cs1, const CachedString& cs2) const {
            return cs1 == cs2;
        }
    };
}

#endif // CACHED_STRING_H

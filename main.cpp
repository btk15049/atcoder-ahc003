#ifndef VSCODE
// clang-format off
 #pragma GCC optimize("Ofast")
 #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")
 #pragma GCC optimize("O3,omit-frame-pointer,inline")
 #pragma GCC optimize("unroll-loops")
// clang-format on
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <map>

namespace constants {

    constexpr int R = 30;
    constexpr int C = 30;
    constexpr int Q = 1000;

    enum direction {
        UP             = 0,
        LEFT           = 1,
        RIGHT          = 2,
        DOWN           = 3,
        DIRECTION_SIZE = 4,
    };
    constexpr int8_t d[]  = {-1, -1, 1, 1};
    constexpr int8_t dr[] = {-1, 0, 0, 1};
    constexpr int8_t dc[] = {0, -1, 1, 0};
    constexpr char ds[]   = "ULRD";

    inline int opposite(int d) { return 3 - d; }
}; // namespace constants


namespace entity {
    struct Point;
    struct Edge;

    struct Point {
        int r, c;

        Point(int r = 0, int c = 0) : r(r), c(c) {}

        inline bool isValid() const {
            if (r < 0 || r >= constants::R) return false;
            if (c < 0 || c >= constants::C) return false;
            return true;
        }

        inline Point neighbor(int i) const {
            return Point(r + constants::dr[i], c + constants::dc[i]);
        }

        template <typename F>
        inline void adjForEach(F f) const {
            for (int i = 0; i < constants::DIRECTION_SIZE; i++) {
                const auto nx = neighbor(i);
                if (!nx.isValid()) continue;
                f(nx);
            }
        }

        Edge getEdge(int dir) const;
    };

    struct Edge {
        Point p;
        int dir;
        Edge(Point p, int dir) : p(p), dir(dir) {}
        inline Edge versus() {
            return Edge(p.neighbor(dir), constants::opposite(dir));
        }
        inline Edge normalize() {
            if (dir >= 2) {
                return versus();
            }
            return *this;
        }
    };

    inline Edge Point::getEdge(int dir) const { return Edge(*this, dir); }

    std::ostream& operator<<(std::ostream& os, Point& p) {
        os << "(" << p.r << ", " << p.c << ")";
        return os;
    }

    std::istream& operator>>(std::istream& is, Point& p) {
        is >> p.r >> p.c;
        return is;
    }

    bool operator<(Point lhs, Point rhs) {
        return ((lhs.r << 10) + lhs.c) < ((rhs.r << 10) + rhs.c);
    }

    bool operator==(Point lhs, Point rhs) {
        return ((lhs.r << 10) + lhs.c) == ((rhs.r << 10) + rhs.c);
    }

    inline int __toValue(Edge e) {
        return (e.p.r << 20) + (e.p.c << 10) + e.dir;
    }

    bool operator<(Edge lhs, Edge rhs) {
        return __toValue(lhs.normalize()) < __toValue(rhs.normalize());
    }

    bool operator==(Edge lhs, Edge rhs) {
        return __toValue(lhs.normalize()) == __toValue(rhs.normalize());
    }


} // namespace entity

using Pair = std::pair<entity::Point, entity::Point>;
std::ostream& operator<<(std::ostream& os, Pair& p) {
    os << "[" << p.first << ", " << p.second << "]";
    return os;
}


namespace history {
    struct Query {
        Pair input;
        std::map<entity::Edge, int> counts;
        std::string output;
        int64_t distance;
        Query(Pair input) : input(input) {}
    };
    std::vector<Query> history;

    void put(Pair p) { history.emplace_back(p); }
    void put(const std::string& output,
             const std::map<entity::Edge, int>& counts, int64_t distance) {
        history.back().output   = output;
        history.back().counts   = counts;
        history.back().distance = distance;
    }
} // namespace history

namespace input {
    std::pair<entity::Point, entity::Point> get(std::istream& is) {
        std::pair<entity::Point, entity::Point> ret;
        is >> ret.first >> ret.second;
        return ret;
    }
} // namespace input

namespace output {
    struct Builder {
        std::string s;
        std::vector<entity::Point> points;
        std::map<entity::Edge, int> count;
        Builder(entity::Point s) { points.emplace_back(s); }

        const entity::Point& getCurrent() const { return points.back(); }

        bool add(int dir) {
            auto nx = points.back().neighbor(dir);
            if (!nx.isValid()) return false;
            count[points.back().getEdge(dir).normalize()]++;
            points.emplace_back(nx);
            s.push_back(constants::ds[dir]);

            return true;
        }

        int64_t fix(std::istream& is, std::ostream& os) {
            os << s << std::endl;
            os.flush();
            int64_t d;
            is >> d;
            return d;
        }
    };


} // namespace output

void solve() {
    for (int q = 0; q < constants::Q; q++) {
        Pair in = input::get(std::cin);
        history::put(in);

        const auto& [s, t] = in;
        output::Builder builder(s);
        while (builder.getCurrent().r > t.r && builder.add(constants::UP))
            ;
        while (builder.getCurrent().r < t.r && builder.add(constants::DOWN))
            ;
        while (builder.getCurrent().c > t.c && builder.add(constants::LEFT))
            ;
        while (builder.getCurrent().c < t.c && builder.add(constants::RIGHT))
            ;
        auto distance = builder.fix(std::cin, std::cout);
        history::put(builder.s, builder.count, distance);
    }
}

#ifndef TEST
int main() { solve(); }
#endif

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
#include <optional>

namespace constants {

    constexpr int R = 30;
    constexpr int C = 30;
    constexpr int Q = 1000;

    constexpr int EDGE_TOTAL = R * C * 2;

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

    std::optional<std::array<std::array<int, constants::C>, constants::R>> h =
        std::nullopt;
    std::optional<std::array<std::array<int, constants::C>, constants::R>> v =
        std::nullopt;
    inline bool hasAns() { return h.has_value(); }
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
                f(nx, constants::opposite(i));
            }
        }

        Edge getEdge(int dir) const;

        inline int getId() const { return r * constants::C + c; }
    };

    struct Edge {
        Point p;
        int dir;
        Edge(Point p, int dir) : p(p), dir(dir) {}
        inline Edge versus() const {
            return Edge(p.neighbor(dir), constants::opposite(dir));
        }
        inline int getId() const {
            const auto e = normalize();
            return (e.p.getId() << 1) + e.dir;
        }
        inline Edge normalize() const {
            if (dir >= 2) {
                return versus();
            }
            return *this;
        }
        inline int getAns() const {
            const auto e = normalize().versus();
            if (e.dir == constants::DOWN) {
                return (*constants::h)[e.p.r][e.p.c];
            }
            else {
                return (*constants::v)[e.p.r][e.p.c];
            }
        }
    };

    inline Edge Point::getEdge(int dir) const { return Edge(*this, dir); }

    std::ostream& operator<<(std::ostream& os, const Point& p) {
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
        std::bitset<constants::EDGE_TOTAL> edges;
        std::string output;
        int64_t distance;
        Query(Pair input) : input(input) {}
    };
    std::vector<Query> history;

    int totalVisits = 0;
    std::array<double, constants::R * constants::C * 2>
        visit; // ucb の 分母に使う
    std::array<int, constants::R * constants::C * 2> useCount;
    std::array<double, constants::R * constants::C * 2> averageSum;


    void init() {
        history.reserve(constants::Q);
        std::fill(visit.begin(), visit.end(), 0.0);
        std::fill(useCount.begin(), useCount.end(), 0);
        std::fill(averageSum.begin(), averageSum.end(), 0.0);
    }


    void put(Pair p) { history.emplace_back(p); }
    void put(const std::string& output, const std::vector<entity::Edge>& edges,
             int64_t distance) {
        history.back().output   = output;
        history.back().distance = distance;

        history.back().edges.reset();
        for (const auto& edge : edges) {
            const int id = edge.getId();
            history.back().edges.set(id);
            visit[id] += 1.0; // / edges.size();
            averageSum[id] += distance / double(edges.size());
            useCount[id]++;
        }
        totalVisits++;
    }
} // namespace history


namespace dijkstra {
    using cost_t = double;
    using id_t   = int;
    struct To {
        cost_t cost;
        int prevDir;
    };
    std::array<To, constants::R * constants::C> table;

    inline double calcUCB1(entity::Edge e) {
        const int id = e.normalize().getId();
        if (history::visit[id] == 0) return 0;
        const double average = history::averageSum[id] / history::useCount[id];
        const double expected =
            std::sqrt(2 * std::log(history::totalVisits) / history::visit[id]);
        return std::max(0.0, average - 500.0 * expected);
    }

    auto dijkstra(Pair st) {
        const auto [s, t] = st;

        for (auto& to : table) {
            to.cost = 1e9; // TODO: 見直す
        }

        using T = std::tuple<cost_t, entity::Point>;
        std::priority_queue<T, std::vector<T>, std::greater<T>> que;

        auto push = [&que](int prevDir, entity::Point next, cost_t c) {
            auto& target = table[next.getId()];
            if (target.cost > c) {
                target.cost    = c;
                target.prevDir = prevDir;
                que.emplace(c, next);
            }
        };

        push(0, t, 0);

        while (!que.empty()) {
            const auto [c, cur] = que.top();
            const double currentCost =
                c; // ラムダで参照するのにコピーが必要らしい、不便
            que.pop();

            if (table[cur.getId()].cost < currentCost) continue;

            cur.adjForEach([&](entity::Point nx, int prevDir) {
                push(prevDir, nx, currentCost + calcUCB1(nx.getEdge(prevDir)));
            });
        }
        std::vector<entity::Edge> edges;

        entity::Point cur = s;
        while (cur.getId() != t.getId()) {
            const int d = table[cur.getId()].prevDir;
            edges.emplace_back(cur, d);
            cur = cur.neighbor(d);
        }

        return std::make_tuple(edges, table[s.getId()].cost);
    }
} // namespace dijkstra

namespace input {
    std::pair<entity::Point, entity::Point> get(std::istream& is) {
        std::pair<entity::Point, entity::Point> ret;
        is >> ret.first >> ret.second;
        return ret;
    }

    void getHV(std::istream& is) {
        constants::h = std::make_optional<decltype(constants::h)::value_type>();
        constants::v = std::make_optional<decltype(constants::v)::value_type>();
        auto& h      = *constants::h;
        auto& v      = *constants::v;
        for (int i = 0; i < constants::R; i++) {
            for (int j = 0; j < constants::C - 1; j++) {
                is >> h[i][j];
            }
        }
        for (int i = 0; i < constants::R - 1; i++) {
            for (int j = 0; j < constants::C; j++) {
                is >> v[i][j];
            }
        }
    }


} // namespace input

namespace output {
    struct Builder {
        std::string s;
        std::vector<entity::Point> points;
        std::vector<entity::Edge> edges;
        Builder(entity::Point s) { points.emplace_back(s); }

        const entity::Point& getCurrent() const { return points.back(); }

        bool add(int dir) {
            auto nx = points.back().neighbor(dir);
            if (!nx.isValid()) return false;
            edges.emplace_back(points.back().getEdge(dir).normalize());
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

void showStat() {
    using namespace std;
    for (int i = 0; i < constants::R - 1; i++) {
        for (int j = 0; j < constants::C - 1; j++) {
            cerr << i << " " << j << " ";
            const auto e =
                entity::Edge(entity::Point(i, j), constants::DOWN).normalize();
            cerr << history::averageSum[e.getId()]
                        / history::useCount[e.getId()]
                 << " ";
            cerr << dijkstra::calcUCB1(e) << " ";
            if (constants::hasAns()) {
                cerr << e.getAns() << " ";
            }

            cerr << endl;
        }
    }
}

void solve() {
    history::init();
    for (int q = 0; q < constants::Q; q++) {
        if (q % 100 == 0) {
            std::cerr << "# turn " << q << ":" << std::endl;
            showStat();
        }
        Pair in = input::get(std::cin);
        history::put(in);

        const auto [edges, cost] = dijkstra::dijkstra(in);

        const auto& [s, t] = in;

        output::Builder builder(s);
        for (auto& it : edges) {
            builder.add(it.dir);
            assert(it.versus().p == builder.getCurrent());
        }

        auto distance = builder.fix(std::cin, std::cout);
        history::put(builder.s, builder.edges, distance);
    }
}

#ifndef TEST
int main() { solve(); }
#endif

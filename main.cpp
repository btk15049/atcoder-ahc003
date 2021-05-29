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
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#ifdef TEST
#    define DBG(x) \
        std::cerr << #x << " = " << (x) << " (L" << __LINE__ << ")" << std::endl
#else
#    define DBG(x) ;
#endif

namespace parameter {
#ifdef UCB1_BIAS_PARAM
    constexpr double UCB1_BIAS = UCB1_BIAS_PARAM;
#else
    constexpr double UCB1_BIAS        = 45.5;
#endif

#ifdef INITIAL_DISTANCE_PARAM
    constexpr double INITIAL_DISTANCE = INITIAL_DISTANCE_PARAM;
#else
    constexpr double INITIAL_DISTANCE = 3088;
#endif

#ifdef ESTIMATE_COUNT_PARAM
    constexpr int ESTIMATE_COUNT = ESTIMATE_COUNT_PARAM;
#else
    constexpr int ESTIMATE_COUNT      = 30;
#endif

#ifdef OLD_BIAS_PARAM
    constexpr double OLD_BIAS = OLD_BIAS_PARAM;
#else
    constexpr double OLD_BIAS         = 0.3;
#endif


} // namespace parameter

namespace xorshift {
    constexpr uint64_t next(uint64_t p) {
        p = p ^ (p << 13);
        p = p ^ (p >> 7);
        return p ^ (p << 17);
    }

    struct Generator {
        uint64_t seed;
        Generator(uint64_t seed = 111111111u) : seed(seed) {}
        inline uint64_t gen() {
            seed = next(seed);
            return seed;
        }
    } _gen;

    inline int64_t getInt(int64_t n) { return _gen.gen() % n; }


} // namespace xorshift

namespace constants {

    constexpr int N = 30;
    constexpr int R = N;
    constexpr int C = N;
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

    uint16_t idMap[constants::DIRECTION_SIZE][constants::R][constants::C];

    uint16_t hIds[constants::R][constants::C - 1];
    uint16_t vIds[constants::C][constants::R - 1];

    struct Edge {
        Point p;
        int dir;
        Edge(int r, int c, int dir) : p(r, c), dir(dir) {}
        Edge(Point p, int dir) : p(p), dir(dir) {}
        inline Edge versus() const {
            return Edge(p.neighbor(dir), constants::opposite(dir));
        }
        inline int getId() const {
            const auto e = normalize();
            return (e.p.getId() << 1) + e.dir;
        }

        inline int getIdFast() const { return idMap[dir][p.r][p.c]; }


        inline Edge normalize() const {
            if (dir >= 2) {
                return versus();
            }
            return *this;
        }
        inline int getAns() const {
            const auto e = normalize().versus();
            if (e.dir == constants::RIGHT) {
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

    struct cww {
        cww() {
            for (int r = 0; r < constants::R; r++) {
                for (int c = 0; c < constants::C; c++) {
                    const Point p(r, c);
                    p.adjForEach([&](Point q, int d) {
                        idMap[d][q.r][q.c] = Edge(q, d).getId();
                    });

                    if (c < constants::C - 1) {
                        hIds[r][c] = Edge(p, constants::RIGHT).getId();
                    }

                    if (r < constants::R - 1) {
                        vIds[c][r] = Edge(p, constants::DOWN).getId();
                    }
                }
            }
        }
    } star;
} // namespace entity

using Pair = std::pair<entity::Point, entity::Point>;
std::ostream& operator<<(std::ostream& os, Pair& p) {
    os << "[" << p.first << ", " << p.second << "]";
    return os;
}


namespace history {
    struct Query {
        Pair input;
        std::vector<entity::Edge> edges;
        std::vector<int> edgeIds;
        std::bitset<constants::EDGE_TOTAL> edgeSet;
        std::string output;
        int64_t distance;
        Query(Pair input) : input(input) {}
    };
    std::vector<Query> queries;

    int totalVisits = 0;
    std::array<double, constants::R * constants::C * 2>
        visit; // ucb の 分母に使う
    std::array<int, constants::R * constants::C * 2> useCount;
    std::vector<int> usedEdgeIds;
    std::array<double, constants::R * constants::C * 2> averageSum;


    void init() {
        queries.reserve(constants::Q);
        std::fill(visit.begin(), visit.end(), 0.0);
        std::fill(useCount.begin(), useCount.end(), 0);
        std::fill(averageSum.begin(), averageSum.end(), 0.0);
    }


    void put(Pair p) { queries.emplace_back(p); }
    void put(const std::string& output, const std::vector<entity::Edge>& edges,
             int64_t distance) {
        queries.back().output   = output;
        queries.back().distance = distance;
        queries.back().edges    = edges;
        queries.back().edgeSet.reset();
        for (const auto& edge : edges) {
            const int id = edge.getIdFast();
            queries.back().edgeIds.push_back(id);
            queries.back().edgeSet.set(id);
            visit[id] += 1.0; // / edges.size();
            averageSum[id] += distance / double(edges.size());
            useCount[id]++;
            if (useCount[id] == 1) usedEdgeIds.push_back(id);
        }
        assert(queries.back().edgeSet.count() > 0u);
        totalVisits++;
    }
} // namespace history

namespace estimate {
    std::array<double, constants::EDGE_TOTAL> estimatedEdgeCost;

    std::array<double, constants::EDGE_TOTAL> old;
    template <typename F>
    inline void edgeForEach(F f) {
        for (int i = 0; i < constants::R; i++) {
            for (int j = 0; j < constants::C - 1; j++) {
                f(entity::Edge(i, j, constants::RIGHT));
            }
        }

        for (int i = 0; i < constants::R - 1; i++) {
            for (int j = 0; j < constants::C; j++) {
                f(entity::Edge(i, j, constants::DOWN));
            }
        }
    }

    namespace forSmoothing {
        double sums[constants::N];
        double squareSums[constants::N];
        int cnts[constants::N];

        // cnt = 0 のときに死ぬので注意
        inline double calcAverage(int bg, int ed) {
            return (sums[ed] - sums[bg]) / (cnts[ed] - cnts[bg]);
        }

        inline double calcVariance(int bg, int ed) {
            const int cnt = cnts[ed] - cnts[bg];
            if (cnt == 0) return 0.0;
            const double average = calcAverage(bg, ed);
            return (squareSums[ed] - squareSums[bg]) / cnt - average * average;
        };

        inline void prepare(const uint16_t* edgeIds) {
            for (int i = 0; i < constants::N - 1; i++) {
                const int id      = edgeIds[i];
                sums[i + 1]       = sums[i];
                squareSums[i + 1] = squareSums[i];
                cnts[i + 1]       = cnts[i];
                if (history::useCount[id] > 0) {
                    cnts[i + 1]++;
                    sums[i + 1] += estimatedEdgeCost[id];
                    squareSums[i + 1] +=
                        estimatedEdgeCost[id] * estimatedEdgeCost[id];
                }
            }
        }

        inline int getSplitPoint(bool isM1 = false) {
            if (isM1) return 0;
            int ret            = constants::N / 2;
            double minVariance = 1e18;

            for (int i = 0; i <= constants::N - 1; i++) {
                const double variance = std::max(
                    calcVariance(0, i), calcVariance(i, constants::N - 1));
                if (minVariance > variance) {
                    minVariance = variance;
                    ret         = i;
                }
            }
            if (xorshift::getInt(2) == 0)
                ret = (ret + xorshift::getInt(constants::N)) / 2;
            return ret;
        };
    } // namespace forSmoothing

    namespace forOptimizeOne {
        std::vector<double> points;

        std::array<std::vector<int>, constants::EDGE_TOTAL> edgeId2HistoryIds;

        inline void prepare() {
            // 最後のクエリ分だけ更新
            for (const auto& e : history::queries.back().edges) {
                edgeId2HistoryIds[e.getIdFast()].push_back(
                    history::queries.size() - 1u);
            }
        }

    } // namespace forOptimizeOne

    // TODO: 高速化
    inline void optimizeOne(int id) {
        using namespace forOptimizeOne;
        points.clear();

        for (const int qId : edgeId2HistoryIds[id]) {
            const auto& query = history::queries[qId];
            double sum        = 0;
            for (const auto& id : query.edgeIds) {
                sum += estimatedEdgeCost[id];
            }
            points.push_back(query.distance - sum);
        }

        // 中央値が見つけられればよいので、乱択 O(n) に変える
        const int midPos = points.size() / 2;
        std::nth_element(points.begin(), points.end(),
                         std::next(points.begin(), midPos));
        estimatedEdgeCost[id] += points[midPos] * 0.1;
        estimatedEdgeCost[id] =
            std::max(1000.0, std::min(9000.0, estimatedEdgeCost[id]));
    };

    inline void smoothing(const uint16_t* ids, bool isM1 = false) {
        using namespace forSmoothing;
        prepare(ids);
        const int mid            = getSplitPoint(isM1);
        constexpr double oldBias = 0.5;

        if (cnts[mid] - cnts[0]) {
            const double average = calcAverage(0, mid);
            for (int i = 0; i < mid; i++) {
                const int id = ids[i];
                if (history::useCount[id] == 0) continue;
                estimatedEdgeCost[id] =
                    oldBias * estimatedEdgeCost[id] + (1 - oldBias) * average;
            }
        }

        if (cnts[constants::N - 1] - cnts[mid]) {
            const double average = calcAverage(mid, constants::N - 1);
            for (int i = mid; i < constants::N - 1; i++) {
                const int id = ids[i];
                if (history::useCount[id] == 0) continue;
                estimatedEdgeCost[id] =
                    oldBias * estimatedEdgeCost[id] + (1 - oldBias) * average;
            }
        }
    }

    void showDetail() {
        if (constants::hasAns()) {
            {
                std::vector<double> errors;
                double errorSum = 0;
                for (const auto& query : history::queries) {
                    double sum = 0;
                    for (const auto& e : query.edgeIds) {
                        sum += estimatedEdgeCost[e];
                    }
                    errors.push_back(abs(sum - query.distance));
                    errorSum += errors.back();
                }
                std::sort(errors.begin(), errors.end());

                for (size_t p : {10, 50, 100, 500, 900}) {
                    if (p < errors.size()) {
                        DBG(errors[p]);
                    }
                }
                if (!errors.empty()) {
                    DBG(errorSum / errors.size());
                }
            }
            double linearError = 0;
            double squareError = 0;
            std::vector<double> errors;
            for (int id : history::usedEdgeIds) {
                const auto e     = entity::Edge((id >> 1) / constants::C,
                                            (id >> 1) % constants::C, id & 1);
                const double err = std::abs(estimatedEdgeCost[id] - e.getAns());
                linearError += err;
                squareError += err * err;
                errors.push_back(estimatedEdgeCost[id] - e.getAns());
            };

            std::sort(errors.begin(), errors.end(),
                      [&](double l, double r) { return l * l < r * r; });

            linearError /= std::max(history::usedEdgeIds.size(), size_t(1));
            squareError /= std::max(history::usedEdgeIds.size(), size_t(1));
            squareError = std::sqrt(squareError);
            DBG(history::usedEdgeIds.size());
            DBG(linearError);
            DBG(squareError);
            if (errors.size() >= 100u) {
                [[maybe_unused]] const double p10 =
                    errors[errors.size() * 10 / 100];
                [[maybe_unused]] const double p30 =
                    errors[errors.size() * 30 / 100];
                [[maybe_unused]] const double p50 =
                    errors[errors.size() * 50 / 100];
                [[maybe_unused]] const double p90 =
                    errors[errors.size() * 90 / 100];
                DBG(p10);
                DBG(p30);
                DBG(p50);
                DBG(p90);
            }
        }
    }


    void setAverage(int turn) {
        const double bias = turn / constants::Q;
        for (int id : history::usedEdgeIds) {
            estimatedEdgeCost[id] = bias * estimatedEdgeCost[id]
                                    + (1.0 - bias) * history::averageSum[id]
                                          / history::useCount[id];
        }
    }


    inline void smoothingAll() {
        std::copy(estimatedEdgeCost.begin(), estimatedEdgeCost.end(),
                  old.begin());
        std::fill(estimatedEdgeCost.begin(), estimatedEdgeCost.end(), 0.0);

        for (const auto& query : history::queries) {
            double sum   = 0;
            const int sz = query.edgeIds.size();
            for (int i = 0; i < sz; i++) {
                sum += old[query.edgeIds[i]];
            }
            for (int i = 0; i < sz; i++) {
                double s = query.distance * (old[query.edgeIds[i]] / sum);
                if (s < 1000.0) {
                    s = 1000;
                }
                else if (s > 9000.0) {
                    s = 9000;
                }
                estimatedEdgeCost[query.edgeIds[i]] += s;
            }
        }
        for (int id : history::usedEdgeIds) {
            estimatedEdgeCost[id] /= history::useCount[id];
        }
    }

    inline double cwwFunction(int x) {
        return (x - 200) * (x - 200) * (x - 1400) / (4 * 1e8) + 1;
    }

    void computeEdgeCost([[maybe_unused]] int turn) {
        forOptimizeOne::prepare();

        setAverage(turn);

        const int threshold = cwwFunction(turn) * parameter::ESTIMATE_COUNT;
        DBG(threshold);

        for (int _ = 0; _ < threshold; _++) {
            smoothingAll();

            // for (int i = 0; i < 100; i++) {
            //     optimizeOne(history::usedEdgeIds[xorshift::getInt(
            //         history::usedEdgeIds.size())]);
            // }

            // O(RC)
            for (int r = 0; r < constants::R; r++) {
                smoothing(entity::hIds[r]);
            }
            for (int c = 0; c < constants::C; c++) {
                smoothing(entity::vIds[c]);
            }
        }
        for (int _ = 0; _ < 5; _++) {
            smoothingAll();
        }

        showDetail();
    }
} // namespace estimate

namespace dijkstra {
    using cost_t = double;
    using id_t   = int;
    struct To {
        cost_t cost;
        int prevDir;
    };

    std::array<To, constants::R * constants::C> table;


    inline double calcUCB1(entity::Edge e) {
        const int id = e.getIdFast();
        if (history::visit[id] == 0) return parameter::INITIAL_DISTANCE;
        const double average =
            estimate::estimatedEdgeCost[id]; // history::averageSum[id] /
                                             // history::useCount[id];
        const double expected =
            std::sqrt(2 * std::log(history::totalVisits) / history::visit[id]);
        return std::max(1000.0, average - parameter::UCB1_BIAS * expected);
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
            cerr << estimate::estimatedEdgeCost[e.getIdFast()] << " ";
            cerr << dijkstra::calcUCB1(e) << " ";
            // cerr << v1::estimatedDistance[e.getId()] << " ";
            if (constants::hasAns()) {
                cerr << e.getAns() << " ";
            }

            cerr << endl;
        }
    }
}

namespace randomWork {
    std::array<std::array<int, constants::C + 2>, constants::R + 2> board;

    constexpr int NG     = -1;
    int currentTimeStamp = 0;
    int ord[24][4];
    void init() {
        {
            int tmp[4], id = 0;
            std::iota(tmp, tmp + 4, 0);
            do {
                ord[id][0] = tmp[0];
                ord[id][1] = tmp[1];
                ord[id][2] = tmp[2];
                ord[id][3] = tmp[3];
                id++;
            } while (std::next_permutation(tmp, tmp + 4));
        }
        currentTimeStamp = 0;
        for (int i = 0; i < constants::R + 2; i++) {
            for (int j = 0; j < constants::C + 2; j++) {
                board[i][j] = currentTimeStamp;
            }
        }

        for (int i : {0, constants::R + 1}) {
            for (int j = 0; j < constants::C + 2; j++) {
                board[i][j] = NG;
            }
        }
        for (int i = 0; i < constants::R + 2; i++) {
            board[i].front() = board[i].back() = NG;
        }
        currentTimeStamp++;
    }

    inline void put(entity::Point p) { board[p.r + 1][p.c + 1] = NG; }

    std::vector<entity::Point> stack;

    inline void update(entity::Point s) {
        currentTimeStamp++;
        stack.clear();

        auto push = [&](entity::Point p, [[maybe_unused]] int _) {
            if (board[p.r + 1][p.c + 1] == NG) {
                return;
            }
            if (board[p.r + 1][p.c + 1] == currentTimeStamp) {
                return;
            }
            board[p.r + 1][p.c + 1] = currentTimeStamp;
            stack.emplace_back(p);
        };

        push(s, 0);
        while (!stack.empty()) {
            const auto p = stack.back();
            stack.pop_back();
            p.adjForEach(push);
        }
    }

    inline bool isOK(entity::Point p) {
        return board[p.r + 1][p.c + 1] == currentTimeStamp;
    }

    void run(output::Builder& builder, entity::Point t) {
        init();
        while (builder.getCurrent().getId() != t.getId()) {
            put(builder.getCurrent());
            update(t);

            const int r = abs(rand()) % 24;
            for (int i = 0; i < 4; i++) {
                const int d  = ord[r][i];
                const auto p = builder.getCurrent().neighbor(d);
                if (isOK(p)) {
                    // std::cerr << p << std::endl;
                    assert(builder.add(d));
                    break;
                }
            }
        }
    }

} // namespace randomWork

void solve() {
    history::init();
    double diffs = 0;
    for (int q = 0; q < constants::Q; q++) {
        DBG("# turn " + std::to_string(q) + ":");
        Pair in = input::get(std::cin);
        history::put(in);

        const auto& [s, t] = in;

        output::Builder builder(s);


        if (q < 0) {
            randomWork::run(builder, t);
        }
        else {
            const auto [edges, cost] = dijkstra::dijkstra(in);
            for (auto& it : edges) {
                builder.add(it.dir);
                assert(it.versus().p == builder.getCurrent());
            }
        }

        auto distance = builder.fix(std::cin, std::cout);
        history::put(builder.s, builder.edges, distance);

        if (q > 100) {
            double sum = 0;
            for (const int id : history::queries.back().edgeIds) {
                sum += estimate::estimatedEdgeCost[id];
            }
            const double diff = sum - distance;
            diffs += abs(diff);
            DBG(diff);
            DBG(distance);
            DBG(diffs / (q + 1));
        }

        if (q != constants::Q - 1) {
            estimate::computeEdgeCost(q + 1);
        }
        // v1::calcEstimateDistance();
    }
}

#ifndef TEST
int main() { solve(); }
#endif

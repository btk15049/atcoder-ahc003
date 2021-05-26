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

#ifdef TEST
#    define DBG(x) \
        std::cerr << #x << " = " << (x) << " (L" << __LINE__ << ")" << std::endl
#else
#    define DBG(x) ;
#endif


namespace other {

    // 少し使いづらいので、行を加算できるようにしておく
    namespace drken {
        using namespace std;
        using D         = double;
        constexpr D EPS = 1e-10;

        // matrix
        template <class T>
        struct Matrix {
            vector<vector<T>> val;
            Matrix() : val() {}
            Matrix(int n, int m, T x = 0) : val(n, vector<T>(m, x)) {}
            void init(int n, int m, T x = 0) { val.assign(n, vector<T>(m, x)); }
            size_t size() const { return val.size(); }
            inline vector<T>& operator[](int i) { return val[i]; }
            void add(vector<T> v) { val.emplace_back(v); }
        };

        template <class T>
        int GaussJordan(Matrix<T>& A, bool is_extended = false) {
            int m = A.size(), n = A[0].size();
            int rank = 0;
            for (int col = 0; col < n; ++col) {
                // 拡大係数行列の場合は最後の列は掃き出ししない
                if (is_extended && col == n - 1) break;

                // ピボットを探す
                int pivot = -1;
                T ma      = EPS;
                for (int row = rank; row < m; ++row) {
                    if (abs(A[row][col]) > ma) {
                        ma    = abs(A[row][col]);
                        pivot = row;
                    }
                }
                // ピボットがなかったら次の列へ
                if (pivot == -1) continue;

                // まずは行を swap
                swap(A[pivot], A[rank]);

                // ピボットの値を 1 にする
                auto fac = A[rank][col];
                for (int col2 = 0; col2 < n; ++col2) A[rank][col2] /= fac;

                // ピボットのある列の値がすべて 0 になるように掃き出す
                for (int row = 0; row < m; ++row) {
                    if (row != rank && abs(A[row][col]) > EPS) {
                        auto fac = A[row][col];
                        for (int col2 = 0; col2 < n; ++col2) {
                            A[row][col2] -= A[rank][col2] * fac;
                        }
                    }
                }
                ++rank;
            }
            return rank;
        }

        template <class T>
        vector<T> linear_equation(Matrix<T> A, vector<T> b) {
            // extended
            int m = A.size(), n = A[0].size();
            Matrix<T> M(m, n + 1);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
                M[i][n] = b[i];
            }
            int rank = GaussJordan(M, true);

            // check if it has no solution
            vector<T> res;
            for (int row = rank; row < m; ++row) {
                if (abs(M[row][n]) > 1) {
                    DBG(row);
                    DBG(M[row][n]);
                    return res;
                }
            }
            // answer
            res.assign(n, 0);
            for (int i = 0; i < rank; ++i) res[i] = M[i][n];
            return res;
        }
    } // namespace drken
} // namespace other

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
    std::vector<Query> queries;

    int totalVisits = 0;
    std::array<double, constants::R * constants::C * 2>
        visit; // ucb の 分母に使う
    std::array<int, constants::R * constants::C * 2> useCount;
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

        queries.back().edges.reset();
        for (const auto& edge : edges) {
            const int id = edge.getId();
            queries.back().edges.set(id);
            visit[id] += 1.0; // / edges.size();
            averageSum[id] += distance / double(edges.size());
            useCount[id]++;
        }
        assert(queries.back().edges.count() > 0u);
        totalVisits++;
    }
} // namespace history


namespace v1 {
    // 最小二乗法で求めた各辺の推定値
    std::array<double, constants::EDGE_TOTAL> estimatedDistance;

    void calcEstimateDistance() {
        other::drken::Matrix<double> X;
        std::vector<double> Y;
        for (int k = 0; k < constants::EDGE_TOTAL; k++) {
            // 左辺
            std::vector<double> x(constants::EDGE_TOTAL, 0.0);
            for (const auto& query : history::queries) {
                if (!query.edges.test(k)) continue;
                for (int j = 0; j < constants::EDGE_TOTAL; j++) {
                    if (query.edges.test(j)) {
                        x[j] += 1.0;
                    }
                }
            }
            double y = 0;
            // 右辺
            for (const auto& query : history::queries) {
                if (query.edges.test(k)) {
                    y += query.distance;
                }
            }
            X.add(x);
            Y.push_back(y);
        }

        const auto A = other::drken::linear_equation(X, Y);
        if (!A.empty()) {
            for (int i = 0; i < constants::EDGE_TOTAL; i++) {
                estimatedDistance[i] = A[i];
            }
        }
    }

} // namespace v1

namespace dijkstra {
    using cost_t = double;
    using id_t   = int;
    struct To {
        cost_t cost;
        int prevDir;
    };
    std::array<To, constants::R * constants::C> table;

    std::array<double, constants::EDGE_TOTAL> estimatedEdgeCost;

    template <typename F>
    void edgeForEach(F f) {
        for (int i = 0; i < constants::R; i++) {
            for (int j = 0; j < constants::C - 1; j++) {
                f(entity::Edge(entity::Point(i, j), constants::RIGHT));
            }
        }

        for (int i = 0; i < constants::R - 1; i++) {
            for (int j = 0; j < constants::C; j++) {
                f(entity::Edge(entity::Point(i, j), constants::DOWN));
            }
        }
    }

    void prepare() {
        int usedEdges = 0;

        std::array<double, constants::EDGE_TOTAL> old;

        edgeForEach([&](entity::Edge e) {
            const int id = e.getId();
            if (history::useCount[id] == 0) return;
            usedEdges++;
            estimatedEdgeCost[id] =
                history::averageSum[id] / history::useCount[id];
            old[id] = estimatedEdgeCost[id];
        });

        std::vector<std::vector<int>> edges;
        for (const auto& query : history::queries) {
            std::vector<int> es;
            for (size_t e = query.edges._Find_first(); e < query.edges.size();
                 e        = query.edges._Find_next(e)) {
                es.push_back(e);
            }
            if (es.size() == 0) continue;
            edges.push_back(es);
        }


        for (int _ = 0; _ < 20; _++) {
            std::fill(estimatedEdgeCost.begin(), estimatedEdgeCost.end(), 0.0);

            for (size_t i = 0; i < edges.size(); i++) {
                double sum = 0;
                for (int e : edges[i]) {
                    sum += old[e];
                }
                for (int e : edges[i]) {
                    const double s =
                        history::queries[i].distance * (old[e] / sum);
                    estimatedEdgeCost[e] += std::min(std::max(0.1, s), 10000.0);
                }
            }
            edgeForEach([&](entity::Edge e) {
                const int id = e.getId();
                if (history::useCount[id] == 0) return;
                estimatedEdgeCost[id] /= history::useCount[id];
                old[id] = estimatedEdgeCost[id];
            });

            auto smoothH = [&](int r, int bg, int ed) {
                double sum = 0;
                int cnt    = 0;
                for (int j = bg; j < ed - 1; j++) {
                    const auto e =
                        entity::Edge(entity::Point(r, j), constants::RIGHT);
                    const int id = e.getId();
                    if (history::useCount[id] == 0) continue;
                    sum += estimatedEdgeCost[id];
                    cnt++;
                }
                for (int j = bg; j < ed - 1; j++) {
                    const auto e =
                        entity::Edge(entity::Point(r, j), constants::RIGHT);
                    const int id = e.getId();
                    if (history::useCount[id] == 0) continue;
                    estimatedEdgeCost[id] = old[id] = sum / cnt;
                }
            };
            auto smoothV = [&](int c, int bg, int ed) {
                double sum = 0;
                int cnt    = 0;
                for (int i = bg; i < ed - 1; i++) {
                    const auto e =
                        entity::Edge(entity::Point(i, c), constants::DOWN);
                    const int id = e.getId();
                    if (history::useCount[id] == 0) continue;
                    sum += estimatedEdgeCost[id];
                    cnt++;
                }
                for (int i = bg; i < ed - 1; i++) {
                    const auto e =
                        entity::Edge(entity::Point(i, c), constants::DOWN);
                    const int id = e.getId();
                    if (history::useCount[id] == 0) continue;
                    estimatedEdgeCost[id] = old[id] = sum / cnt;
                }
            };
            if (_ < 15) {
                for (int i = 0; i < constants::R; i++) {
                    smoothH(i, 0, constants::C / 2);
                    smoothH(i, constants::C / 2, constants::C);
                }
                for (int j = 0; j < constants::C; j++) {
                    smoothV(j, 0, constants::R / 2);
                    smoothV(j, constants::R / 2, constants::R);
                }
            }
        }

        if (constants::hasAns()) {
            double linearError = 0;
            double squareError = 0;
            std::vector<double> errors;
            edgeForEach([&](entity::Edge e) {
                const int id = e.getId();
                if (history::useCount[id] == 0) return;
                const double err = std::abs(estimatedEdgeCost[id] - e.getAns());
                linearError += err;
                squareError += err * err;
                errors.push_back(estimatedEdgeCost[id] - e.getAns());
            });

            std::sort(errors.begin(), errors.end(),
                      [&](double l, double r) { return l * l < r * r; });

            linearError /= std::max(usedEdges, 1);
            squareError /= std::max(usedEdges, 1);
            squareError = std::sqrt(squareError);
            DBG(usedEdges);
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
            // for (size_t i = 0; i < edges.size(); i++) {
            //     double sum = 0;
            //     for (int e : edges[i]) {
            //         sum += old[e];
            //     }
            //     const double diff = history::queries[i].distance - sum;
            //     // DBG(diff / edges[i].size());
            // }
        }
    }

    inline double calcUCB1(entity::Edge e) {
        const int id = e.normalize().getId();
        if (history::visit[id] == 0) return 3000;
        const double average =
            estimatedEdgeCost[id]; // history::averageSum[id] /
                                   // history::useCount[id];
        const double expected =
            std::sqrt(2 * std::log(history::totalVisits) / history::visit[id]);
        return std::max(0.0, average - 5.0 * expected);
    }

    auto dijkstra(Pair st) {
        prepare();
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
            cerr << dijkstra::estimatedEdgeCost[e.getId()] << " ";
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

        // v1::calcEstimateDistance();
    }
}

#ifndef TEST
int main() { solve(); }
#endif

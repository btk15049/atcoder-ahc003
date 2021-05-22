#define TEST

#include "main.cpp"

void dijkstraTest() {
    std::fill(history::averageSum.begin(), history::averageSum.end(), 1e5);
    std::fill(history::useCount.begin(), history::useCount.end(), 1);
    std::fill(history::visit.begin(), history::visit.end(), 1.0);
    history::totalVisits              = 30 * 29 * 2;
    history::averageSum[entity::Edge(entity::Point(1, 1), constants::DOWN)
                            .normalize()
                            .getId()] = 10.0;
    history::averageSum[entity::Edge(entity::Point(2, 1), constants::DOWN)
                            .normalize()
                            .getId()] = 20.0;
    history::averageSum[entity::Edge(entity::Point(3, 1), constants::RIGHT)
                            .normalize()
                            .getId()] = 10.0;
    history::averageSum[entity::Edge(entity::Point(1, 1), constants::RIGHT)
                            .normalize()
                            .getId()] = 30.0;
    history::averageSum[entity::Edge(entity::Point(1, 2), constants::DOWN)
                            .normalize()
                            .getId()] = 10.0;
    history::averageSum[entity::Edge(entity::Point(2, 2), constants::DOWN)
                            .normalize()
                            .getId()] = 5.0;
    const auto [edges, cost] =
        dijkstra::dijkstra(Pair(entity::Point(1, 1), entity::Point(3, 2)));
    const double expected = 40 - sqrt(2 * std::log(2 * 30 * 29)) * 3;

    assert(std::abs(cost - expected) < 1e-5);
}

int main() { dijkstraTest(); }

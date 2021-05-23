echo "" > scores.log; 

g++ debug.cpp -std=c++17 -o a.out

cd tools

for f in (ls in/)
    cargo run --release --bin tester in/$f ../a.out ans/$f 2>log/$f.log | tee scores.log 
end

python3 tot.py <scores.log

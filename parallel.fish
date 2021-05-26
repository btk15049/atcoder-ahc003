echo "" > scores.log; 

g++ main.cpp -std=c++17 -o a.out

cd tools

for f in (ls log/)
    rm log/$f
end

for f in (ls scores/)
    rm scores/$f
end

set cnt 0
for f in (ls in/)
    # echo $f
    cargo run --release --bin tester in/$f ../a.out ans/$f 2>log/$f.log | tee scores/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "10"
        sleep 2
        set cnt 0
        python3 tot.py ./scores
    end
end


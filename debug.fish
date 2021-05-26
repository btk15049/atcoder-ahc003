echo "" > scores.log; 

g++ debug.cpp -std=c++17 -o a.out

cd tools

for f in (ls log/)
    rm log/$f
end

for f in (ls scores1/)
    rm scores1/$f
end

for f in (ls scores2/)
    rm scores2/$f
end

set cnt 0
for f in (ls in1/)
    # echo $f
    cargo run --release --bin tester in1/$f ../a.out ans/$f 2>log/$f.log | tee scores1/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "10"
        sleep 2
        set cnt 0
        python3 tot.py ./scores1
    end
end

if test "$cnt" != "0"
    sleep 2
    set cnt 0
    python3 tot.py ./scores1
end

set cnt 0
for f in (ls in2/)
    # echo $f
    cargo run --release --bin tester in2/$f ../a.out ans/$f 2>log/$f.log | tee scores2/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "10"
        sleep 2
        set cnt 0
        python3 tot.py ./scores2
    end
end

if test "$cnt" != "0"
    sleep 2
    set cnt 0
    python3 tot.py ./scores2
end


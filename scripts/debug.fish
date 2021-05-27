set REPO_ROOT (dirname (dirname (readlink -m (status --current-filename))))

set P $argv[1]
if test "$P" = ""
    set P 10
end

set WORKDIR /tmp/(dbus-uuidgen)
mkdir $WORKDIR
echo "debug.fish: workdir is $WORKDIR"
mkdir $WORKDIR/scores1
mkdir $WORKDIR/scores2

set BIN $WORKDIR/a.out
g++ $REPO_ROOT/debug.cpp -std=c++17 -Wall -Wextra -o $BIN


cd $REPO_ROOT/tools
set cnt 0

for f in (ls in2/)
    cargo run --release --bin tester in2/$f $BIN ans/$f 2>log/$f.log | tee $WORKDIR/scores2/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "$P"
        sleep 2
        set cnt 0
        python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores2
    end
end

for f in (ls in1/)
    cargo run --release --bin tester in1/$f $BIN ans/$f 2>log/$f.log | tee $WORKDIR/scores1/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "$P"
        sleep 2
        set cnt 0
        python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores1
    end
end

sleep 2

echo "M=1:" (python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores1)
echo "M=2:" (python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores2)



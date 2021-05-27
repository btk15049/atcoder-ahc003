set REPO_ROOT (dirname (dirname (readlink -m (status --current-filename))))

set P $argv[1]
if test "$P" = ""
    set P 10
end

set S $argv[2]
if test "$S" = ""
    set S 2
end

set WORKDIR /tmp/(dbus-uuidgen)
mkdir $WORKDIR
echo "parallel.fish: workdir is $WORKDIR"
mkdir $WORKDIR/logs
mkdir $WORKDIR/scores

set BIN $WORKDIR/a.out
g++ $REPO_ROOT/main.cpp -std=c++17 -Wall -Wextra -o $BIN

cd $REPO_ROOT/tools

set cnt 0
for f in (ls in/)
    cargo run --release --bin tester in/$f $BIN 2>$WORKDIR/logs/$f.log | tee $WORKDIR/scores/$f &
    set cnt (math $cnt + 1)
    if test "$cnt" = "$P"
        sleep $S
        set cnt 0
        python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores
    end
end

sleep $S
echo "score:" (python3 $REPO_ROOT/py-tools/score.py $WORKDIR/scores)

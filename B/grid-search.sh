spars=( "0.9" "0.99" )
#ngpus=( 4 8 16 )
for spar in "${spars[@]}"
#for ngpu in "${ngpus[@]}"
do
    ./scripts/killp2p.sh
    sleep 5
    sparsity=$spar ./dr.sh
    #ngpu=$ngpu ./dr.sh
    sleep 2000
done

dev_id=3
nohup ./nvml_samples -device=$dev_id -si=50 -output=power.log 1>/dev/null 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=$dev_id ./volumeRender -file=data/ref_volume.ppm -size=1024
sleep 5
killall nvml_samples

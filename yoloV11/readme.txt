Code to split a dataset by 50: 
ls -v *.jpg | awk 'NR % 50 == 0' | xargs -I{} cp {} /home/mainubuntu/Desktop/Repositories/yoloV11/annotated_data

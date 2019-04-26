> train.list
COUNT=-1
for folder in $1/*
do
    COUNT=$[$COUNT + 1]
    for setFolder in "$folder"/*
    do
		if [ "$folder"/"train" = "$setFolder" ] ; then
			for imagesFolder in "$setFolder"/*
			do
		        	echo "$imagesFolder" $COUNT >> train.list
        		done
		fi
    done
done


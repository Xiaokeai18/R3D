> test.list
COUNT=-1
for folder in $1/*
do
	#if [ "$1"/"BENEFIT" = "$folder" ] || [ "$1"/"ABOUT" = "$folder" ] ; then
	    COUNT=$[$COUNT + 1]
	    for setFolder in "$folder"/*
		do
		if [ "$folder"/"train" != "$setFolder" ] ; then
		  
			for imagesFolder in "$setFolder"/*
			    do
				echo "$imagesFolder" $COUNT >> test.list

			    done
		fi
		done
	#fi
done


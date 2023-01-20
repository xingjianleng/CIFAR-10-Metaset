classes=(airplane bird cat deer dog frog horse sedan ship suv truck)
for i in "${classes[@]}"
do
	echo -e "\nDownloading label: $i"
	python3 "auto_download.py" "$i"
done

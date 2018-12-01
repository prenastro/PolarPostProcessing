#This code takes two arguments - source urls file where urls to be downloaded are stored per line and an output directory where the downloaded files are expected to be stored

url_filename="$1"
output_filelocation="$2"
while read -r line; do
    url="$line"
    wget -P $output_filelocation -t 1 -T 5 $url # -P for output folder, -t for number of retries and -T for timeout
    echo "$url"
done < "$url_filename"

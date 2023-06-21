filelist_txt="tocommit"
while IFS= read -r filename; do
echo $filename
filename=$(echo "$filename" | tr -d '[:space:]' )
echo $filename
git add "$filename"
done < "$filelist_txt"

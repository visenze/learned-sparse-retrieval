token=$(grep "FURY_PUSH_AUTH=" ~/.dltk.config | awk 'NR>1{print $1}' RS="=")
[[ -z "$token" ]] && { echo "GemFury push token is invalid in ~/.dltk.config" ; }
python setup.py sdist --formats=gztar
file_name=dist/`python setup.py --fullname`.tar.gz
curl -F package=@${file_name} https://${token}@push.fury.io/visenze/
rm -rf dist
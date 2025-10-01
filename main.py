import pkg_resources, os

def get_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

for dist in pkg_resources.working_set:
    size = get_size(dist.location + "/" + dist.project_name.replace("-", "_"))
    print(f"{dist.project_name:40} {dist.version:10} {size/1024/1024:.2f} MB")

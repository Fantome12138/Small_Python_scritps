from pathlib import Path
import glob

def check_file(file):
    """
    用在train.py和test.py文件中  检查本地有没有这个文件
    检查相关文件路径能否找到文件 并返回文件名
    Search/download file (if necessary) and return path
    """
    file = str(file)  # convert to str()
    # 如果传进来的是文件或者是’‘, 直接返回文件名str
    if Path(file).is_file() or file == '':  # exists
        return file
    # 如果传进来的以 'http:/' 或者 'https:/' 开头的url地址, 就下载
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        # urllib.parse: 解析url  .unquote: 对url进行解码   file: 要下载的文件名
        # '%2F' to '/', split https://url.com/file.txt?auth
        file = Path(urllib.parse.unquote(file)).name.split('?')[0]
        print(f'Downloading {url} to {file}...')
        # 使用torch.hub.download_url_to_file从url地址上中下载文件名为file的文件
        torch.hub.download_url_to_file(url, file)
        # 检查是否下载成功
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        # 返回下载的文件名
        return file
    else:
        # 否则, 传进来的就是当前项目下的一个全局路径 查找匹配的文件名 返回第一个
        # glob.glob: 匹配当前项目下的所有项目 返回所有符合条件的文件files
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        # 返回第一个匹配到的文件名
        return files[0]  # return file


print(check_file('/test_code'))


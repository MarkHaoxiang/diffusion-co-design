import os


def scp(source: str, path: str, target: str):
    os.system('scp "%s:%s" "%s"' % (source, path, target))

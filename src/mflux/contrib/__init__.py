# Namespace package declaration using pkgutil
# This allows multiple contrib packages to coexist
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

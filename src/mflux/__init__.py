# Namespace package declaration using pkgutil
# This allows mflux and mflux-schedulers to coexist
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

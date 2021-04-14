#!/usr/bin/env python3
import abc


class Module(abc.ABC):
    @abc.abstractmethod
    def get_params(self) -> dict:
        raise NotImplementedError

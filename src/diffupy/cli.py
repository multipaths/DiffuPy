# -*- coding: utf-8 -*-

"""Command line interface."""

import logging

import click

log = logging.getLogger(__name__)


@click.group(help='PathMe')
def main():
    """Run PathMe."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


"""Diffupy"""


@main.group()
def diffupy():
    """Run experiment."""


if __name__ == '__main__':
    main()

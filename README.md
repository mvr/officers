Officers
========

A CUDA program to calculate the Grundy values of Officers as fast as
possible. Officers is the only unsolved "single-digit octal game",
conjecturally, the values are eventually periodic.

The first few values are

    0, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 0, 3, 4, 2, 1, 3, 2, 1, ...

appearing as OEIS sequence [A046695](https://oeis.org/A046695).

Grossman calculated 140 trillion values with no sign of periodicity
yet! If I can get this properly optimised, the goal is to push that a
bit further. Right now this gets 1 million positions per second, which
is frankly unacceptable.


Rules
-----

Officers is a two-player take-and-break game played with stacks of
coins. A turn consists of removing a coin from a stack and leaving the
remainder in either one or two stacks. In particular, taking away a
single coin is not a valid move. The winner is the last player who can
make a move.

References
----------

* Berlekamp, Elwyn R.; Conway, John H.; Guy, Richard K. Winning ways for your mathematical plays. Vol. 1. Second edition. A K Peters, Ltd., Natick, MA, 2001. xx+276 pp. ISBN: 1-56881-130-6
* Gangolli, A.; Plambeck, T. [A note on periodicity in some octal games](https://link.springer.com/content/pdf/10.1007/BF01254294.pdf). Internat. J. Game Theory 18 (1989), no. 3, 311--320 
* Grossman, J. P. [Searching for periodicity in officers](https://library.slmath.org/books/Book70/files/1016.pdf). Games of no chance 5, 373--385, Math. Sci. Res. Inst. Publ., 70, Cambridge Univ. Press, Cambridge, 2019. 
* Flammenkamp, A. [Sprague-Grundy Values of Octal-Games](https://wwwhomes.uni-bielefeld.de/achim/grundy.html). 2021

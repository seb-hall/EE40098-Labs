#import "@preview/plotst:0.2.0": *

#import "resources/EE40098-References.yml"

#set page(
    paper: "a4",
    margin: (x: 1.25cm, top: 1.5cm, bottom: 1.5cm),
    columns: 2,
    header:  context {
        if(counter(page).get().at(0) != 1) [
            *EE40098 Computational Intelligence - Coursework B*
            #h(1fr)
            #counter(page).display(
                "1/1",
                both: true,
            )
        ]
    }
)

#set text(
    size: 11pt
)

#set par(
    justify: true,
    leading: 0.52em,
)

#set heading(
    numbering: "1."
)

#place(
    top + center,
    float: true,
    scope: "parent",
    {
        text(17pt)[
            *EE40098 Computational Intelligence - Coursework B* \
        ]

        text(13pt)[
            Seb Hall #link("mailto:samh25@bath.ac.uk"), 21st November 2025\
            Department of Electronic & Electrical Engineering, University of Bath \
        ]
    }
)

// MARK: INTRO
= Introduction

Genetic algorithms (GAs) are a type of iterative algorithm based on biological evolution. 

// MARK: EX 1
= Exercise 1

_Implementation of a simple genetic algorithm to search for a target value._

A simple genetic algorithm was created in Python to search for a number in the shortest number of iterations. This was achieved with an object-oriented approach that defined an 'Individual' class to represent a candidate solution, and a 'Population' class to manage individuals and the genetic process.

Three genetic processes were implemented:

1. *Selection* - a proportion of the most fit individuals are selected to remain in the population.
2. *Mutation* - some individuals have their genes modified randomly to introduce genetic diversity.
3. *Reproduction* - pairs of individuals are combined to produce offspring with genes from both.

In this implementation, 


// MARK: EX 2
= Exercise 2
_Analysis of the genetic algorithm created in exercise 1._

// MARK: EX 3
= Exercise 3
_Implementation a stop condition for the algorithm created in exercise 1._

// MARK: EX 4
= Exercise 4
_Optimising a _

// MARK: EX 5
= Exercise 5

// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)
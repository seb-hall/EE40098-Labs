#import "@preview/plotst:0.2.0": *

#import "resources/EE40098-References.yml"

#set page(
    paper: "a4",
    margin: (x: 1.25cm, top: 1.5cm, bottom: 1.5cm),
    columns: 2,
    header:  context {
        if(counter(page).get().at(0) != 1) [
            *EE40098 Computational Intelligence - Coursework A*
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
            *EE40098 Computational Intelligence - Coursework A* \
        ]

        text(13pt)[
            Seb Hall #link("mailto:samh25@bath.ac.uk"), 24th October 2025\
            Department of Electronic & Electrical Engineering, University of Bath \
        ]
    }
)

// MARK: INTRO

// MARK: EX 1
= Exercise 1 - Logical Operators

Truth tables, state space graphs and neural network weights and biases for the AND, NAND, OR and XOR logical operators are shown below. In the graphs, red indicates value of 0 and blue indicates value of 1.

== AND Operator

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "AND Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
                align(center, //Align starts here
                    table(
                        columns: (auto, auto, auto),
                        inset: 7.5pt,
                        align: horizon + center,
                        stroke: (col, row) => (
                            bottom: if row == 0 { 1pt } else { 0pt },
                            right: if col == 1 { 1pt } else { 0pt },
                        ),
                        table.header(
                            [*A*], [*B*], [*AND(A,B)*],
                        ),
                        [0], [0], [0],
                        [1], [0], [0],
                        [0], [1], [0],
                        [1], [1], [1],
                    )
                )
            )
        ) <and-truth-table>
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/and_plot.png", width: 100%)
            ]),
            caption: [AND State Space],

        )  <and-state-space>
    ] 
    )
])

#figure(
    image("resources/and_network_diagram.png", width: 100%),
    caption: [AND network diagram, showing \ weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -1.5).],
)  <and-network-diagram>

== NAND Operator
 
#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "NAND Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
                align(center, //Align starts here
                    table(
                        columns: (auto, auto, auto),
                        inset: 7.5pt,
                        align: horizon + center,
                        stroke: (col, row) => (
                            bottom: if row == 0 { 1pt } else { 0pt },
                            right: if col == 1 { 1pt } else { 0pt },
                        ),
                        table.header(
                            [*A*], [*B*], [*NAND(A,B)*],
                        ),
                        [0], [0], [1],
                        [1], [0], [1],
                        [0], [1], [1],
                        [1], [1], [0],
                    )
                )
            )
        ) <nand-truth-table>
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/nand_plot.png", width: 100%)
            ]),
            caption: [NAND State Space],

        )  <nand-state-space>
    ] 
    )
])

#figure(
    image("resources/nand_network_diagram.png", width: 100%),
    caption: [NAND network diagram, showing \ weights (w#sub[1] = -1, w#sub[2] = -1) and bias (b = 1.5).],
)  <nand-network-diagram>

== OR Operator

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "OR Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
                align(center, //Align starts here
                    table(
                        columns: (auto, auto, auto),
                        inset: 7.5pt,
                        align: horizon + center,
                        stroke: (col, row) => (
                            bottom: if row == 0 { 1pt } else { 0pt },
                            right: if col == 1 { 1pt } else { 0pt },
                        ),
                        table.header(
                            [*A*], [*B*], [*OR(A,B)*],
                        ),
                        [0], [0], [0],
                        [1], [0], [1],
                        [0], [1], [1],
                        [1], [1], [1],
                    )
                )
            )
        ) <or-truth-table>
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/or_plot.png", width: 100%)
            ]),
            caption: [OR State Space],

        )  <or-state-space>
    ] 
    )
])

#figure(
    image("resources/or_network_diagram.png", width: 100%),
    caption: [OR network diagram, showing \ weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -0.5).],
)  <or-network-diagram>

== XOR Operator
 

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "XOR Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
                align(center, //Align starts here
                    table(
                        columns: (auto, auto, auto),
                        inset: 7.5pt,
                        align: horizon + center,
                        stroke: (col, row) => (
                            bottom: if row == 0 { 1pt } else { 0pt },
                            right: if col == 1 { 1pt } else { 0pt },
                        ),
                        table.header(
                            [*A*], [*B*], [*XOR(A,B)*],
                        ),
                        [0], [0], [0],
                        [1], [0], [1],
                        [0], [1], [1],
                        [1], [1], [0],
                    )
                )
            )
        ) <xor-truth-table>
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/xor_plot.png", width: 100%)
            ]),
            caption: [XOR State Space],

        )  <xor-state-space>
    ] 
    )
])

#figure(
    image("resources/xor_network_diagram.png", width: 100%),
    caption: [XOR network diagram, consisting of a hidden layer with AND and OR weights, followed by an output layer, with weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -0.5).],
)  <xor-network-diagram>


// MARK: EX 2
= Exercise 2 - Implementation of a Neural Network for MNIST

A perceptron-based neural network can be implemented to classify images into discrete categories. For this exercise, a network was implemented to classify MNIST datasets; the first a set of handwritten numeric digits, and the second a collection of items of clothing.

== Neural Network 1 - Numeric Dataset

=== Network Structure

#figure(
    image("resources/ann_diagram.png", width: 110%),
    caption: [Artificial Neural Network (ANN) Structure for Classifying MNIST Digits.],
)  <ann-diagram>

An ANN was implemented as shown in @ann-diagram, consisting of an input layer with 784 input nodes corresponding to the 28x28 grayscale pixels in each image, a hidden layer with a variable number of nodes, and an output layer with 10 nodes corresponding to the 10 output digit classes (0-9).

=== Hyperparameter Search

In order to optimise the performance of the network, two parameters were explored in detail - the learning rate and number of hidden nodes.
This was achieved with a Monte Carlo search method, which involves defining a parameter space, and evaluating random combinations of parameters within this space to identify an optimal region of combinations (see @monte-carlo-100k-search).

#figure(
    image("resources/parameters_search_100k.png", width: 110%),
    caption: [Monte Carlo Search for Hyperparameters of Hidden Nodes and Learning Rate.],
)  <monte-carlo-100k-search>

For the MNIST digit dataset, a search was performed on the reduced train and test datasets for 100,000 iterations, with learning rates between 0.01 and 1.0, and hidden layers between  and 1000. This was a wide search space, so a large number of samples were required to find optimal parameters. The output of this search was represented as a scatter plot, displaying the test performance of each sample as a colour gradient, from 0-90%. This image was then edited to increase contrast and find a clearer optimal region, shown in @monte-carlo-100k-search-contrasted.

#figure(
    image("resources/parameters_search_100k-contrasted.png", width: 90%),
    caption: [High Contrast Monte Carlo Output with Optimal Point Selected.],
)  <monte-carlo-100k-search-contrasted>

Taking a sample in the center of the largest optimal region produced a point with 254 hidden nodes and a learning rate of 0.11.

=== Training Iteration Search

After selecting optimal values for hidden layer nodes and learning rate, the network was trained with a variety of test iterations to indentify the optimal number of training cycles. The results of this are shown below in @training-iterations-graph.

#figure(
    image("resources/parameters_search_100k.png", width: 110%),
    caption: [Test Performance vs Training Iterations for Optimal Hyperparameters.],
)  <training-iterations-graph>

=== Full Training and Testing

After identifying optimal hyperparameters, the network was trained and tested on the full MNIST dataset. The results for this are shown below:

== Neural Network 2 - Fashion Dataset

The second dataset used was the Fashion MNIST dataset, which consists of 28x28 grayscale images of clothing items instead of numeric digits.

=== Hyperparameter Search

A similar approach was taken to identify optimal hyperparameters for this dataset, however the reduced size training dataset was significantly larger than before (1000 vs 100 samples), so each iteration of the Monte Carlo search would take longer to complete. Therefore, a new search was performed with 25,000 iterations and a smaller parameter space; learning rates between 0.01 and 0.5, and hidden layers between 1 and 500. The results from this are shown below: 

#figure(
    image("resources/fashion_parameters_search_25k.png", width: 110%),
    caption: [Monte Carlo Search for Hyperparameters of Hidden Nodes and Learning Rate (Fashion MNIST).],
)  <training-fashion-iterations>


#figure(
    image("resources/fashion_parameters_search_25k-selected.png", width: 90%),
    caption: [High Contrast Monte Carlo Output with Optimal Point Selected (Fashion MNIST).],
)  <training-fashion-iterations-spotted>

Hidden nodes: 214/506 = 0.42. 0.42*400 = 168. 168 + 300 = 468.
Learning Rate: 290/421 = 0.69. 1 - 0.69 = 0.31. 0.31*0.1 = 0.031.

Hidden nodes: 430/509 = 0.84. 0.84*500 = 420.
Learning Rate = 400/423 = 0/94. 1-0.94 = 0.06. 0.06*0.5 = 0.03.




=== Training Iteration Search
As before, a search for the optimal number of training iterations was performed, with the results shown below:


=== Full Training and Testing

// MARK: CONCLUSION
= Conclusion

// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)


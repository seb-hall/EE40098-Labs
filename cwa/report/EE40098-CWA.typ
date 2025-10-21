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

Perceptrons@perceptron can be used to perform logical operations on binary and discrete data inputs. Truth tables, state space graphs, and neural network weights and biases for the AND, NAND, OR and XOR logical operators are shown below. In the graphs, red indicates a value of 0 and blue indicates a value of 1.

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
    caption: [XOR network diagram, consisting of a hidden layer with AND and OR weights, followed by an output layer, with weights (w#sub[1] = -1, w#sub[2] = 1) and bias (b = -0.5).],
)  <xor-network-diagram>


// MARK: EX 2
= Exercise 2 - Implementation of a Neural Network for MNIST

A perceptron-based neural network can be implemented to classify images into discrete categories. For this exercise, a network was implemented to classify MNIST@mnist datasets; the first a set of handwritten numeric digits, and the second a collection of items of clothing.

== Neural Network 1 - Numeric Dataset

=== Network Structure

#figure(
    image("resources/ann_diagram.png", width: 110%),
    caption: [Artificial Neural Network (ANN) Structure for Classifying MNIST Digits.],
)  <ann-diagram>

An artificial neural network (ANN) was implemented as shown in @ann-diagram, consisting of an input layer with 784 input nodes corresponding to the 28x28 grayscale pixels in each image, a hidden layer with a variable number of nodes, and an output layer with 10 nodes corresponding to the 10 output digit classes (0-9).

=== Hyperparameter Search

To optimise the performance of the network, two parameters were explored in detail - the learning rate and number of hidden nodes.
This was achieved with a Monte Carlo search method@monte-carlo, which involves defining a parameter space, and evaluating random combinations of parameters within this space to produce an optimal region (see @monte-carlo-100k-search).

#figure(
    image("resources/parameters_search_100k.png", width: 110%),
    caption: [Monte Carlo Search for Hyperparameters of Hidden Nodes and Learning Rate.],
)  <monte-carlo-100k-search>

A search was performed on reduced train and test datasets for increased speed. 100,000 iterations were taken with learning rates between 0.01 and 1.0, and hidden layers between 1 and 1000. This was a wide search space, so a large number of samples were required to find optimal parameters. The output of this search was represented as a scatter plot, displaying the test performance of each sample as a colour gradient, from 0-90%. This image was then edited to increase contrast and find a clearer optimal region, shown in @monte-carlo-100k-search-contrasted.

#figure(
    image("resources/parameters_search_100k-contrasted.png", width: 75%),
    caption: [High Contrast Monte Carlo Output with \ Optimal Point Selected.],
)  <monte-carlo-100k-search-contrasted>

Taking a sample in the center of the largest optimal region produced a point with *254 hidden nodes* and a *learning rate of 0.11*.

=== Full Dataset Performance

After selecting optimal values for hidden layer nodes and learning rate, the network was trained over a number of iterations on the full dataset. The results for this are shown below in @training-iterations-graph.

#figure(
    image("resources/training_iterations_search_25_0.png", width: 110%),
    caption: [Test Performance vs Training Iterations for Optimal Hyperparameters.],
)  <training-iterations-graph>

The network reached a high level of performance after three training iterations, achieving *97.4% accuracy*. Beyond this, the performance fluctuated slightly before gradually decreasing (possibly due to overfitting of the data).

== Neural Network 2 - Fashion Dataset

The second dataset used was the Fashion MNIST dataset, which consists of 28x28 grayscale images of clothing items instead of numeric digits. A structure identical to that shown in @ann-diagram was used, due to the images being the same size, and the number of distinct classes also being 10.

=== Hyperparameter Search

A similar approach was taken to identify optimal hyperparameters for this dataset, however the reduced size training dataset was significantly larger than before (1000 vs 100 samples), so each training iteration took significantly longer to complete. Therefore, a new Monte Carlo search was performed with 25,000 iterations and a smaller parameter space; learning rates between 0.01 and 0.5, and hidden layers between 1 and 500. The results from this are shown below: 

#figure(
    image("resources/fashion_parameters_search_25k.png", width: 110%),
    caption: [Monte Carlo Search for Hyperparameters of Hidden Nodes and Learning Rate (Fashion MNIST).],
)  <training-fashion-iterations>


#figure(
    image("resources/fashion_parameters_search_25k-selected.png", width: 75%),
    caption: [High Contrast Monte Carlo Output with Optimal Point Selected (Fashion MNIST).],
)  <training-fashion-iterations-spotted>

As with the previous dataset, the output image was edited for increased contrast to determine an optimal region, shown in @training-fashion-iterations-spotted. A sample in the center of the optimal region was selected to give a resulting count of *420 hidden nodes* and a *learning rate of 0.03*.

=== Full Dataset Performance

As before, the hyperparameters identified with the Monte Carlo search were then applied to training of the full dataset over a number of iterations (@training-fashion-iterations-graph).

#figure(
    image("resources/fashion_training_iterations_search_25_0.png", width: 110%),
    caption: [Test Performance vs Training Iterations for Optimal Hyperparameters (Fashion MNIST).],
)  <training-fashion-iterations-graph>

The performance took longer to stabilise than previously, eventually reaching a peak of *87.5% accuracy* after 9 training iterations.

=== Comparison to Digit Dataset

The final performance of the fashion network was significantly lower than that of the digit network, with error rates of 12.5% and 2.6% respectively, an increase of approximately 5x. 

An additional observation is that the fashion network performed best with a higher number of hidden nodes and a lower learning rate compared to the digit network. 
This may be indicative of a dataset with more complex features, requiring more nodes to represent them.

// MARK: CONCLUSION
== Possible Improvements

A wide range of network architectures could be explored to provide higher performance, such as different activation functions or multiple hidden layers. Convolutional Neural Networks (CNNs) are also frequently used for image classification tasks, and have additional features such as pooling and convolutional layers. These are especially effective at extracting complex features, and would likely be the most beneficial for the fashion dataset which struggled with a simpler ANN architecture.

// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)
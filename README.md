# Multi-Lable Residual Network

I wanted to show how good software development practices can be applied to a
state of the art convolutional model. Typically, these models are defined in
ugly config files and the business logic is hard to find.

This model uses the simple scikit-learn API and provides a clean interface
that lets the user(me) clearly see what choices are being made to construct
the model. For example, the image processing options and output layer activation
are important and shouldn't be hidden deep a script.

# Acceleration prediction
----
This project is a continuation of work done here: https://github.com/minimum-LaytonC/DMproject
(though no data, and little to no code from that project is used here)

We attempt to predict next state acceleration of the human wrist from current observations of acceleration, rotation, magnetic field, and electrical activity of the biceps and triceps. Our data comes from normal human activity, collected with a Raspberry Pi taped to my arm powered by a battery in my pocket.

In the previous project simpler algorithms were applied to a more limited dataset. The previous dataset included only acceleration and a single EMG on the biceps, and activity was limited to a fixed arm position with only one axis of motion. Here activity is unrestricted.

Welcome to view and use our code, which includes code related to sparsification and quality assessment. Then, in the auction, there is independent reverse auction code based on reputation values, which can be run directly.
In our code design:

Regarding Top-K sparsification, you can use different sparsity rates to observe its performance and communication costs, where communication costs are replaced by the bits of the gradient parameters.

Regarding the quality assessment mechanism, after training is complete, corresponding images are generated, and you can observe the differences in similarity between different types of vehicles.

Regarding the reverse auction algorithm, you can run the auction file independently. It will randomly generate reputation values and bids. You can observe from the runtime results that it satisfies “individual rationality,” “budget feasibility,” and “authenticity.”
## Usage
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST 

GRSRB needs to be downloaded separately and saved in the data folder under TRAINING_DATA.

`python federated_learning.py`

will run the Federated Learning experiment specified in  

`python auction.py`

will run the specified reverse auction experiment. 

`federated_learning.json`.

You can specify:

### Task
- `"dataset"` : Choose from `["mnist","fashionmnist","traffic_sign"]`
- `"net"` : Choose from `["cnn","lstm",  "vgg11s"]`

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients
- `"iterations"` : Total number of training iterations



### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.

## Options
- `--schedule` : specify which batch of experiments to run, defaults to "main"





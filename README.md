# p2pdemo
composing model-based proofs of skill using NFTs on Ethereum

inspiration:
* gnark: https://github.com/Consensys/gnark-tests/blob/main/solidity/solidity_groth16_test.go
* EZKL: https://github.com/zkonduit/ezkl
* RISC ZERO: https://github.com/risc0/risc0/tree/v0.16.0/examples/factors
* ZK-DTP: https://github.com/only4sim/ZK-DTP
* Axiom: https://github.com/axiom-crypto/uniswap-v3-oracles/tree/main/contracts

steps: 
* pull historical trading data for a given ETH wallet address from Uniswap
* (we parse Etherscan, but preferably done with Axiom)
* preprocess json (clean, reformat)
* using simple data simulation script in python to test model first
* create and train pytorch model:
    * engineer two features: 
        * buy/sell x token; 
        * before/after y date;
    * multi-classification labeling: fish; shark; neither;
    * split dataset into training and testing
    * train
    * evaluate
* export .onnx file (open-source neural network standard)
* use EZKL for ZKP system:
    * generate SRS
    * generate proving, verification keys
    * generate proof for given network
    * verify proof
    * post verification contract on EVM
* user calls contract to verify proof on-chain + mint nft
* on-chain poker table can use set membership to compose proofs and guarantee threshold for fish

![alt text](./p2p%20demo%20outline.png)
from web3 import Web3
import json

# Connect to Ganache RPC
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Make sure connection is successful
if not w3.is_connected():
    raise Exception("❌ Could not connect to Ganache at http://127.0.0.1:7545")

# Use the first unlocked account from Ganache
w3.eth.default_account = w3.eth.accounts[0]

# Contract address from your Hardhat deployment
contract_address = Web3.to_checksum_address("0x4E879570512f58414B395796fea0B9F37E12986C")

# ABI from your Hardhat script output (as JSON string)
contract_abi = json.loads("""
[
  {
    "type": "event",
    "anonymous": false,
    "name": "Report",
    "inputs": [
      {"type": "address", "name": "target", "indexed": true},
      {"type": "string", "name": "vulnerability", "indexed": false},
      {"type": "uint256", "name": "timestamp", "indexed": false}
    ]
  },
  {
    "type": "function",
    "name": "logFinding",
    "constant": false,
    "payable": false,
    "inputs": [
      {"type": "address", "name": "target"},
      {"type": "string", "name": "vulnerability"}
    ],
    "outputs": []
  }
]
""")

# Create contract object
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Function to log vulnerabilities on-chain
def log_vulnerability(target_address, vulnerability_type):
    try:
        tx_hash = contract.functions.logFinding(target_address, vulnerability_type).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Logged {vulnerability_type} for {target_address}.")
        print(f"Tx Hash: {receipt.transactionHash.hex()}")
    except Exception as e:
        print("❌ Failed to log vulnerability:", e)

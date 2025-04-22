// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditLog {
    event Report(address indexed target, string vulnerability, uint timestamp);

    function logFinding(address target, string memory vulnerability) public {
        emit Report(target, vulnerability, block.timestamp);
    }
}

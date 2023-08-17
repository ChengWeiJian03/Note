JCC

|    指令     |                     EFLAG                      |                          解释                           |
| :---------: | :--------------------------------------------: | :-----------------------------------------------------: |
|    JZ/JE    |                  ZF=1（相等）                  |               jump if zero;jump if equal                |
|   JNZ/JNE   |                 ZF=0（不相等）                 |           jump if not zero;jump if not equal            |
|     JS      |               JS=1（结果为负数）               |                      jump if sign                       |
|     JNS     |               JS=0（结果为正数）               |                    jump if not sign                     |
|   JP/JPE    |             PF=1（1的个数为偶数）              |                  jump if Parity (Even)                  |
|   JNP/JPO   |             PF=0（1的个数为奇数）              |                jump if not parity (odd)                 |
|     JO      |                  OF=1（溢出）                  |                    jump if overflow                     |
|     JNO     |                 OF=0（不溢出）                 |                  jump if not overflow                   |
| JC/JB/JNAE  |                  CF=1（借位）                  |   jump if carry;jump if below;jump if not above equal   |
| JNC/JNB/JAE |                 CF=0（不借位）                 | jump if not carry;jump if not below;jump if above equal |
|   JBE/JNA   |           ZF=0或CF=1（相等或者借位）           |          jump if below equal;jump if not above          |
|   JNBE/JA   |         ZF=0或者CF=0（相等或者不借位）         |          jump if not below equal;jump if above          |
|   JL/JNGE   |            SF!=OF（为负数且不溢出）            |         jump if less;jump if not greater equal          |
|   JNL/JGE   |           SF=OF（为整数并且不溢出）            |         jump if not less;jump if greater equal          |
|   JLE/JNG   | SF != OF 或 ZF=1（为负数且不溢出或者结果相等） |         jump if less equal;jump if not greater          |
|   JNLE/JG   |  SF=0F 且 ZF=0（为正数并且没溢出，且不相等）   |         jump if not less equal;jump if greater          |


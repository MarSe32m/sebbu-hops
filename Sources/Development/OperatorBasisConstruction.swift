//
//  OperatorBasisConstruction.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 4.5.2026.
//

import SebbuScience
import SebbuCollections
import DequeModule

func hilbertSchmidtInnerProduct(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>) -> Complex<Double> {
    var result: Complex<Double> = .zero
    for i in 0..<A.elements.count {
        result += A.elements[i].conjugate * B.elements[i]
    }
    return result
}

func orthonormalize(_ O: Matrix<Complex<Double>>) -> Matrix<Complex<Double>> {
    O / hilbertSchmidtInnerProduct(O, O).real.squareRoot()
}

func commutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>, into: inout Matrix<Complex<Double>>) {
    A.dot(B, into: &into)
    B.dot(A, multiplied: Complex(-1), addingInto: &into)
}

func projection(_ O: Matrix<Complex<Double>>, onto basis: [Matrix<Complex<Double>>]) -> Matrix<Complex<Double>> {
    var result: Matrix<Complex<Double>> = .zeros(rows: O.rows, columns: O.columns)
    for B in basis {
        result += hilbertSchmidtInnerProduct(B, O) * B
    }
    return result
}

func constructBasis(H: Matrix<Complex<Double>>, L: Matrix<Complex<Double>>...) -> [Matrix<Complex<Double>>] {
    constructBasis(H: H, L: L)
}

@discardableResult
func addIfIndependent(
    _ O: Matrix<Complex<Double>>,
    to basis: [Matrix<Complex<Double>>],
    tolerance: Double = 1e-10
) -> (Bool, Matrix<Complex<Double>>) {
    var R = O

    for B in basis {
        R.subtract(B, multiplied: hilbertSchmidtInnerProduct(B, R))
    }
    // For numerical stability
    for B in basis {
        R.subtract(B, multiplied: hilbertSchmidtInnerProduct(B, R))
    }

    let oNorm = O.frobeniusNorm
    let rNorm = R.frobeniusNorm

    if rNorm > tolerance * max(1.0, oNorm) {
        R.divide(by: rNorm)
        return (true, R)
    }

    return (false, R)
}

func constructBasis(
    H: Matrix<Complex<Double>>,
    L: [Matrix<Complex<Double>>]
) -> [Matrix<Complex<Double>>] {
    struct Index: Hashable {
        let LDaggerIndex: Int
        let O1Index: Int
        let O2Index: Int
    }
    
    var HIndex = 0
    var LIndex = 0
    var LDaggersTested: Set<Index> = Set()
    let LDaggers = L.map { $0.conjugateTranspose }
    var basis: [Matrix<Complex<Double>>] = []
    var LDaggerOs: [[Matrix<Complex<Double>>]] = .init(repeating: [], count: LDaggers.count)
    for O in L {
        print("Initially added")
        let (shouldAdd, R) = addIfIndependent(O, to: basis)
        if shouldAdd {
            basis.append(R)
            for i in LDaggers.indices {
                LDaggerOs[i].append(LDaggers[i].dot(R))
            }
        }
    }

    let maxDimension = H.rows * H.columns
    var scratch: Matrix<Complex<Double>> = .identity(rows: H.rows)
    var scratch2: Matrix<Complex<Double>> = .identity(rows: H.rows)
    while true {
        var didFindNewBasisElement = false
        if HIndex < basis.count {
            commutator(H, basis[HIndex], into: &scratch)
            let (shouldAdd, R) = addIfIndependent(scratch, to: basis)
            if shouldAdd {
                print("Found in [H, O]", basis.count)
                basis.append(R)
                for i in LDaggers.indices {
                    LDaggerOs[i].append(LDaggers[i].dot(R))
                }
            }
            didFindNewBasisElement = shouldAdd || didFindNewBasisElement
            if basis.count >= maxDimension { return basis }
            HIndex += 1
        }
        if LIndex < basis.count {
            for Li in L {
                commutator(Li, basis[LIndex], into: &scratch)
                let (shouldAdd, R) = addIfIndependent(scratch, to: basis)
                if shouldAdd {
                    print("Found in [L, O]", basis.count)
                    basis.append(R)
                    for i in LDaggers.indices {
                        LDaggerOs[i].append(LDaggers[i].dot(R))
                    }
                }
                didFindNewBasisElement = shouldAdd || didFindNewBasisElement
                if basis.count >= maxDimension { return basis }
            }
            LIndex += 1
        }
        print(basis.count * basis.count * LDaggers.count)
        for (i1, _) in basis.enumerated() {
            for (i2, O2) in basis.enumerated() {
                for (iLD, _) in LDaggers.enumerated() {
                    let index = Index(LDaggerIndex: iLD, O1Index: i1, O2Index: i2)
                    if LDaggersTested.contains(index) {
                        continue
                    }
                    LDaggersTested.insert(index)
                    commutator(LDaggerOs[iLD][i1], O2, into: &scratch2)
                    let (shouldAdd, R) = addIfIndependent(scratch2, to: basis)
                    if shouldAdd {
                        print("Found in [LDagO, O]", basis.count)
                        basis.append(R)
                        for i in LDaggers.indices {
                            LDaggerOs[i].append(LDaggers[i].dot(R))
                        }
                    }
                    didFindNewBasisElement = shouldAdd || didFindNewBasisElement
                    if basis.count >= maxDimension { return basis }
                }
            }
        }
        if !didFindNewBasisElement || basis.count >= maxDimension { break }
    }

    return basis
}

func constructBasisFast(
    H: Matrix<Complex<Double>>,
    L: [Matrix<Complex<Double>>],
    tolerance: Double = 1e-10
) -> [Matrix<Complex<Double>>] {
    enum WorkItem {
        case commutatorWithFixed(Int, Int)
        // fixed operator index, basis index

        case nonlinear(Int, Int, Int)
        // L index, alpha index, beta index
    }
    
    let fixedOperators = [H] + L
    let LDaggers = L.map { $0.conjugateTranspose }

    var basis: [Matrix<Complex<Double>>] = []
    var LDaggerTimesBasis: [[Matrix<Complex<Double>>]] =
        Array(repeating: [], count: L.count)

    var work: Deque<WorkItem> = Deque()

    let maxDimension = H.rows * H.columns
    var scratch: Matrix<Complex<Double>> = H
    func tryAppendBasisElement(_ O: Matrix<Complex<Double>>) {
        if basis.count >= maxDimension {
            return
        }

        let originalNorm = O.frobeniusNorm
        if originalNorm == 0 {
            return
        }

        var R = O

        // Modified Gram-Schmidt
        for B in basis {
            R.subtract(B, multiplied: hilbertSchmidtInnerProduct(B, R))
        }

        // Reorthogonalize once for numerical stability
        for B in basis {
            R.subtract(B, multiplied: hilbertSchmidtInnerProduct(B, R))
        }

        let residualNorm = R.frobeniusNorm

        guard residualNorm > tolerance * max(1.0, originalNorm) else {
            return
        }

        basis.append(R / residualNorm)
        let newIndex = basis.count - 1

        // Precompute L† O_new once
        for ell in L.indices {
            LDaggerTimesBasis[ell].append(LDaggers[ell].dot(basis[newIndex]))
        }

        // Add [H, O_new]
        work.append(.commutatorWithFixed(0, newIndex))

        // Add [L_i, O_new]
        for i in L.indices {
            work.append(.commutatorWithFixed(i + 1, newIndex))
        }

        // Add [L_i† O_alpha, O_beta] involving the new element.
        //
        // Since newIndex is the newest basis element, all older pairs have
        // already been scheduled. So only schedule pairs where at least one
        // index is newIndex.
        for j in 0...newIndex {
            for ell in L.indices {
                work.append(.nonlinear(ell, j, newIndex))

                if j != newIndex {
                    work.append(.nonlinear(ell, newIndex, j))
                }
            }
        }
    }

    // Initial seed basis
    for O in L {
        tryAppendBasisElement(O)
    }
    
    while let item = work.popFirst() {
        if basis.count >= maxDimension {
            break
        }

        switch item {
        case let .commutatorWithFixed(fixedIndex, beta):
            commutator(fixedOperators[fixedIndex], basis[beta], into: &scratch)

        case let .nonlinear(ell, alpha, beta):
            let A = LDaggerTimesBasis[ell][alpha]
            commutator(A, basis[beta], into: &scratch)
        }

        tryAppendBasisElement(scratch)
    }

    return basis
}

private func printMatrix<T>(_ M: Matrix<T>) {
    for i in 0..<M.rows {
        print("|", terminator: " ")
        for j in 0..<M.columns {
            if j != M.columns - 1 {
                print(M[i,j], terminator: ", ")
            } else {
                print(M[i, j], terminator: " ")
            }
        }
        print("|")
    }
}

func printBasis(_ basis: [Matrix<Complex<Double>>]) {
    for (i, B) in basis.enumerated() {
        print("Basis element", i)
        printMatrix(B)
        print()
    }
}

func _tensor(baseDimension: Int, subSystems: Int, O: Matrix<Complex<Double>>, at: Int) -> Matrix<Complex<Double>> {
    var result: Matrix<Complex<Double>> = O
    for i in 0..<subSystems where i != at {
        if i < at {
            result = .identity(rows: baseDimension).kronecker(result)
        } else {
            result = result.kronecker(.identity(rows: baseDimension))
        }
    }
    return result
}

func basisTest() {
    var H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let XX: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    var basis = constructBasis(H: H, L: XX)
    printBasis(basis)
    
    H = .init(elements: [.zero, .zero, .zero, Complex(3.21)], rows: 2, columns: 2)
    basis = constructBasis(H: H, L: sigmaMinus)
    printBasis(basis)
    
    func pow(_ base: Int, _ exponent: Int) -> Int {
        var result = 1
        for _ in 0..<exponent {
            result *= base
        }
        return result
    }
    
    let subsystems = 9
    let dim = 2
    let totalDimension = pow(dim, subsystems)
    var _H: Matrix<Complex<Double>> = .zeros(rows: totalDimension, columns: totalDimension)
    for i in 0..<subsystems {
        _H += _tensor(baseDimension: dim, subSystems: subsystems, O: H, at: i)
    }
    var Ls: [Matrix<Complex<Double>>] = []
    for i in 0..<subsystems {
        Ls.append(_tensor(baseDimension: dim, subSystems: subsystems, O: XX, at: i))
        //Ls.append(_tensor(baseDimension: dim, subSystems: subsystems, O: sigmaMinus, at: i))
    }
    let time = ContinuousClock().measure {
        //basis = constructBasis(H: _H, L: Ls)
        basis = constructBasis(H: _H, L: Ls)
    }
    //printBasis(basis)
    print(basis.count, time)
}

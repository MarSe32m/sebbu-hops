//
//  HEOMHierarchy.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 1.11.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms
import Dispatch

@usableFromInline
internal struct AuxiliaryDensityOperator: Sendable {
    @usableFromInline
    let index: Int
    
    @usableFromInline
    let leftPositiveNeighbourIndices: [(coefficient: Complex<Double>, neighbourIndex: Int)]
    @usableFromInline
    let rightPositiveNeighbourIndices: [(coefficient: Complex<Double>, neighbourIndex: Int)]
    @usableFromInline
    let leftNegativeNeighbourIndices: [(coefficient: Complex<Double>, neighbourIndex: Int)]
    @usableFromInline
    let rightNegativeNeighbourIndices: [(coefficient: Complex<Double>, neighbourIndex: Int)]
    
    @usableFromInline
    let dampingFactor: Complex<Double>
    
    @inlinable
    init(index: Int, kVectorPair: (left: [Int], right: [Int]), G: [Complex<Double>], W: [Complex<Double>], leftPositiveNeighbourIndices: [(component: Int, neighbourIndex: Int)], rightPositiveNeighbourIndices: [(component: Int, neighbourIndex: Int)], leftNegativeNeighbourIndices: [(component: Int, neighbourIndex: Int)], rightNegativeNeighbourIndices: [(component: Int, neighbourIndex: Int)]) {
        self.index = index
        var _dampingFactor: Complex<Double> = .zero
        for i in kVectorPair.left.indices {
            _dampingFactor += Double(kVectorPair.left[i]) * W[i]
            _dampingFactor += Double(kVectorPair.right[i]) * W[i].conjugate
        }
        self.dampingFactor = _dampingFactor
        self.leftPositiveNeighbourIndices = leftPositiveNeighbourIndices.map { (component, neighbourIndex) in
            (Double(kVectorPair.left[component] + 1).squareRoot() * .sqrt(G[component]), neighbourIndex)
            //(Complex(1.0), neighbourIndex)
        }
        self.rightPositiveNeighbourIndices = rightPositiveNeighbourIndices.map { (component, neighbourIndex) in
            (Double(kVectorPair.right[component] + 1).squareRoot() * .sqrt(G[component].conjugate), neighbourIndex)
            //(Complex(1.0), neighbourIndex)
        }
        self.leftNegativeNeighbourIndices = leftNegativeNeighbourIndices.map { (component, neighbourIndex) in
            (Double(kVectorPair.left[component]).squareRoot() * .sqrt(G[component]), neighbourIndex)
            //(Double(kVectorPair.left[component]) * G[component], neighbourIndex)
        }
        self.rightNegativeNeighbourIndices = rightNegativeNeighbourIndices.map { (component, neighbourIndex) in
            (Double(kVectorPair.right[component]).squareRoot() * .sqrt(G[component].conjugate), neighbourIndex)
            //(Double(kVectorPair.right[component]) * G[component].conjugate, neighbourIndex)
        }
    }
    
    @inline(__always)
    @inlinable
    @_transparent
    internal func step(H: Matrix<Complex<Double>>, L: Matrix<Complex<Double>>, LDagger: Matrix<Complex<Double>>, rates: [Double], jumpOperators: [(O: Matrix<Complex<Double>>, ODagger: Matrix<Complex<Double>>, ODaggerO: Matrix<Complex<Double>>)], currentStates: [Matrix<Complex<Double>>], scratch: inout Matrix<Complex<Double>>, into: inout Matrix<Complex<Double>>) {
        let rho = currentStates[index]
        // Hamiltonian
        commutator(H, rho, multiplied: -.i, into: &into)
        // Lindblad jump operators
        assert(rates.count == jumpOperators.count)
        for (rate, jumpOperator) in zip(rates, jumpOperators) where rate != .zero {
            lindbladian(rate: rate, rho: rho, O: jumpOperator.O, ODagger: jumpOperator.ODagger, ODaggerO: jumpOperator.ODaggerO, scratch: &scratch, addingInto: &into)
        }
        // Damping factor
        into.add(rho, multiplied: -dampingFactor)
        for (coefficient, neighbourIndex) in rightPositiveNeighbourIndices {
            commutator(L, currentStates[neighbourIndex], multiplied: coefficient, addingInto: &into)
        }
        for (coefficient, neighbourIndex) in leftPositiveNeighbourIndices {
            commutator(currentStates[neighbourIndex], LDagger, multiplied: coefficient, addingInto: &into)
        }
        for (coefficient, neighbourIndex) in leftNegativeNeighbourIndices {
            L.dot(currentStates[neighbourIndex], multiplied: coefficient, addingInto: &into)
        }
        for (coefficient, neighbourIndex) in rightNegativeNeighbourIndices {
            currentStates[neighbourIndex].dot(LDagger, multiplied: coefficient, addingInto: &into)
        }
    }
}

@inline(__always)
@inlinable
@_transparent
func commutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>, multiplied: Complex<Double>, into: inout Matrix<Complex<Double>>) {
    A.dot(B, multiplied: multiplied, into: &into)
    B.dot(A, multiplied: -multiplied, addingInto: &into)
}

@inline(__always)
@inlinable
@_transparent
func commutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>, multiplied: Complex<Double>, addingInto into: inout Matrix<Complex<Double>>) {
    A.dot(B, multiplied: multiplied, addingInto: &into)
    B.dot(A, multiplied: -multiplied, addingInto: &into)
}

@inline(__always)
@inlinable
@_transparent
func lindbladian(rate: Double, rho: Matrix<Complex<Double>>, O: Matrix<Complex<Double>>, ODagger: Matrix<Complex<Double>>, ODaggerO: Matrix<Complex<Double>>, scratch: inout Matrix<Complex<Double>>, into: inout Matrix<Complex<Double>>) {
    O.dot(rho, multiplied: Complex(rate), into: &scratch)
    scratch.dot(O, into: &into)
    ODaggerO.dot(rho, multiplied: Complex(-rate * 0.5), addingInto: &into)
    rho.dot(ODaggerO, multiplied: Complex(-rate * 0.5), addingInto: &into)
}

@inline(__always)
@inlinable
@_transparent
func lindbladian(rate: Double, rho: Matrix<Complex<Double>>, O: Matrix<Complex<Double>>, ODagger: Matrix<Complex<Double>>, ODaggerO: Matrix<Complex<Double>>, scratch: inout Matrix<Complex<Double>>, addingInto into: inout Matrix<Complex<Double>>) {
    O.dot(rho, multiplied: Complex(rate), into: &scratch)
    scratch.dot(O, addingInto: &into)
    ODaggerO.dot(rho, multiplied: Complex(-rate * 0.5), addingInto: &into)
    rho.dot(ODaggerO, multiplied: Complex(-rate * 0.5), addingInto: &into)
}

public struct HEOMHierarchy2: Sendable {
    /// Coupling operator
    public let L: Matrix<Complex<Double>>
    
    /// Dimension of the system
    public let dimension: Int
    
    // The auxiliary density operators (hierarchy)
    @usableFromInline
    internal let ADOS: [AuxiliaryDensityOperator]
    
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], depth: Int) {
        self = .init(dimension: dimension, L: L, G: G, W: W) { kVector in
            kVector.reduce(0, +) <= depth
        }
    }
    
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], truncationCondition: ([Int]) -> Bool) {
        self.dimension = dimension
        self.L = L
        self.ADOS = HEOMHierarchy2._constructADOs(G: G, W: W, truncationCondition: truncationCondition)
    }
    
    @inlinable
    @inline(__always)
    public func solve(start: Double = .zero, end: Double, initialState: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, lindbladians: [(Double, Matrix<Complex<Double>>)] = [], stepSize: Double = 0.01) -> (tSpace: [Double], densityMatrix: [Matrix<Complex<Double>>]) {
        solve(start: start, end: end, initialState: initialState, H: {_, _ in H }, lindbladians: lindbladians.map { rate, O in ({_ in rate}, O) }, stepSize: stepSize)
    }
    
    @inlinable
    @inline(__always)
    public func solve(start: Double = .zero, end: Double, initialState: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, lindbladians: [((Double) -> Double, Matrix<Complex<Double>>)], stepSize: Double = 0.01) -> (tSpace: [Double], densityMatrix: [Matrix<Complex<Double>>]) {
        solve(start: start, end: end, initialState: initialState, H: {_, _ in H }, lindbladians: lindbladians, stepSize: stepSize)
    }
    
    @inlinable
    public func solve(start: Double = .zero, end: Double, initialState: Matrix<Complex<Double>>, H: (Double, Matrix<Complex<Double>>) -> Matrix<Complex<Double>>, lindbladians: [((Double) -> Double, Matrix<Complex<Double>>)] = [], stepSize: Double = 0.01) -> (tSpace: [Double], densityMatrix: [Matrix<Complex<Double>>]) {
        precondition(initialState.rows == self.dimension)
        precondition(initialState.columns == self.dimension)
        var initialHierarchy: [Matrix<Complex<Double>>] = .init(repeating: .zeros(rows: dimension, columns: dimension), count: self.ADOS.count)
        initialHierarchy[0] = initialState
        var resultCache: Deque<[Matrix<Complex<Double>>]> = .init(repeating: initialHierarchy, count: 4)
        let LDagger = L.conjugateTranspose
        var rates: [Double] = .init(repeating: 0, count: lindbladians.count)
        let jumpOperators = lindbladians.map { _, O in
            (O, O.conjugateTranspose, O.conjugateTranspose.dot(O))
        }
        var scratch: Matrix<Complex<Double>> = initialState
        return withoutActuallyEscaping(H) { H in
            var solver = RK4Solver(initialState: initialHierarchy, t0: start, dt: stepSize) { t, currentStates in
                nonisolated(unsafe) var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                for i in rates.indices {
                    rates[i] = lindbladians[i].0(t)
                }
                let H = H(t, currentStates[0])
//                result.withUnsafeMutableBufferPointer { resultBuffer in
//                    nonisolated(unsafe) let result = resultBuffer
//                    DispatchQueue.concurrentPerform(iterations: currentStates.count) { i in
//                        ADOS[i].step(H: H, L: L, LDagger: LDagger, currentStates: currentStates, into: &result[i])
//                    }
//                }
                for i in currentStates.indices {
                    ADOS[i].step(H: H, L: L, LDagger: LDagger, rates: rates, jumpOperators: jumpOperators, currentStates: currentStates, scratch: &scratch, into: &result[i])
                }
                return result
            }
            
            var tSpace: [Double] = []
            var densityMatrix: [Matrix<Complex<Double>>] = []
            while solver.t <= end {
                let (t, states) = solver.step()
                tSpace.append(t)
                var rho: Matrix<Complex<Double>> = .zeros(rows: dimension, columns: dimension)
                rho.copyElements(from: states[0])
                densityMatrix.append(rho)
            }
            return (tSpace, densityMatrix)
        }
    }
    
    @inlinable
    internal static func _constructADOs(G: [Complex<Double>], W: [Complex<Double>], truncationCondition: ([Int]) -> Bool) -> [AuxiliaryDensityOperator] {
        let kVectorPairs = _generateKVectorPairs(components: G.count, truncationCondition: truncationCondition)
        let positiveNeighbours = _generatePositiveNeighbours(kVectorPairs: kVectorPairs)
        let negativeNeighbours = _generateNegativeNeighbours(kVectorPairs: kVectorPairs)
        var result: [AuxiliaryDensityOperator] = []
        for (index, kVectorPair) in kVectorPairs.enumerated() {
            let positiveNeighbours = positiveNeighbours[index]
            let negativeNeighbours = negativeNeighbours[index]
            let ado = AuxiliaryDensityOperator(index: index, kVectorPair: kVectorPair, G: G, W: W, leftPositiveNeighbourIndices: positiveNeighbours.left, rightPositiveNeighbourIndices: positiveNeighbours.right, leftNegativeNeighbourIndices: negativeNeighbours.left, rightNegativeNeighbourIndices: negativeNeighbours.right)
            result.append(ado)
        }
        return result
    }
    
    
    @inlinable
    internal static func _generateKVectorPairs(components: Int, truncationCondition: ([Int]) -> Bool) -> [(left: [Int], right: [Int])] {
        var kVectors: [[Int]] = []
        for sum in 0... {
            let partitions = sum.partitions(maxTerms: components).map { partition in
                if partition.count == components { return partition }
                precondition(partition.count < components)
                return partition + [Int](repeating: 0, count: components - partition.count)
            }
            var validKTuplesFound = false
            for partition in partitions {
                for permutation in partition.lazy.uniquePermutations().filter(truncationCondition) {
                    kVectors.append(permutation)
                    validKTuplesFound = true
                }
            }
            if !validKTuplesFound { break }
        }
        var result: [([Int], [Int])] = []
        for leftVector in kVectors {
            for rightVector in kVectors {
                result.append((leftVector, rightVector))
            }
        }
        return result
//        return result.sorted { lhs, rhs in
//            lhs.0.reduce(0, +) < rhs.0.reduce(0, +) && lhs.1.reduce(0, +) < rhs.1.reduce(0, +)
//        }
    }
    
    @usableFromInline
    struct NeighbourKey: Hashable {
        @usableFromInline
        let left: [Int]
        @usableFromInline
        let right: [Int]
        
        @inlinable
        init(_ leftRight: ([Int], [Int])) {
            self.left = leftRight.0
            self.right = leftRight.1
        }
    }
    
    @inlinable
    internal static func _generatePositiveNeighbours(kVectorPairs: [([Int], [Int])]) -> [(left: [(component: Int, neighbourIndex: Int)], right: [(component: Int, neighbourIndex: Int)])] {
        var index: [NeighbourKey:Int] = [:]
        index.reserveCapacity(kVectorPairs.count)
        for (i, kVectorPair) in kVectorPairs.enumerated() {
            index[NeighbourKey(kVectorPair)] = i
        }
        var result: [(left: [(Int, Int)], right: [(Int, Int)])] = []
        result.reserveCapacity(kVectorPairs.count)
        for kVectorPair in kVectorPairs {
            var leftNeighbours: [(Int, Int)] = []
            var rightNeighbours: [(Int, Int)] = []
            var neighbour = kVectorPair
            for i in 0..<kVectorPair.0.count {
                neighbour.0[i] += 1; defer { neighbour.0[i] -= 1}
                if let neighbourIndex = index[NeighbourKey(neighbour)] {
                    leftNeighbours.append((i, neighbourIndex))
                }
            }
            for i in 0..<kVectorPair.1.count {
                neighbour.1[i] += 1; defer { neighbour.1[i] -= 1}
                if let neighbourIndex = index[NeighbourKey(neighbour)] {
                    rightNeighbours.append((i, neighbourIndex))
                }
            }
            result.append((leftNeighbours, rightNeighbours))
        }
        return result
    }
    
    @inlinable
    internal static func _generateNegativeNeighbours(kVectorPairs: [([Int], [Int])]) -> [(left: [(component: Int, neighbourIndex: Int)], right: [(component: Int, neighbourIndex: Int)])] {
        var index: [NeighbourKey:Int] = [:]
        index.reserveCapacity(kVectorPairs.count)
        for (i, kVectorPair) in kVectorPairs.enumerated() {
            index[NeighbourKey(kVectorPair)] = i
        }
        var result: [(left: [(Int, Int)], right: [(Int, Int)])] = []
        result.reserveCapacity(kVectorPairs.count)
        for kVectorPair in kVectorPairs {
            var leftNeighbours: [(Int, Int)] = []
            var rightNeighbours: [(Int, Int)] = []
            var neighbour = kVectorPair
            for i in 0..<kVectorPair.0.count {
                neighbour.0[i] -= 1; defer { neighbour.0[i] += 1}
                if let neighbourIndex = index[NeighbourKey(neighbour)] {
                    leftNeighbours.append((i, neighbourIndex))
                }
            }
            for i in 0..<kVectorPair.1.count {
                neighbour.1[i] -= 1; defer { neighbour.1[i] += 1}
                if let neighbourIndex = index[NeighbourKey(neighbour)] {
                    rightNeighbours.append((i, neighbourIndex))
                }
            }
            result.append((leftNeighbours, rightNeighbours))
        }
        return result
    }
}

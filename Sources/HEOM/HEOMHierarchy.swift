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

public struct HEOMHierarchy: Sendable {
    /// Coupling operator
    public let L: Matrix<Complex<Double>>
    
    /// Dimension of the system
    public let dimension: Int
    
    // "Louvillian" matrix
    @usableFromInline
    internal let B: CSRMatrix<Complex<Double>>
    
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], lindbladians: [(Double, Matrix<Complex<Double>>)], depth: Int) {
        self = .init(dimension: dimension, L: L, G: G, W: W, lindbladians: lindbladians) { kVector in
            kVector.reduce(0, +) <= depth
        }
    }
    
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], lindbladians: [(Double, Matrix<Complex<Double>>)], truncationCondition: ([Int]) -> Bool) {
        self.dimension = dimension
        self.L = L
        self.B = HEOMHierarchy._constructBMatrix(dimension: dimension, L: L, G: G, W: W, lindbladians: lindbladians, truncationCondition: truncationCondition)
    }
    
    @inlinable
    @inline(__always)
    public func solve(start: Double = .zero, end: Double, initialState: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, stepSize: Double = 0.01) -> (tSpace: [Double], densityMatrix: [Matrix<Complex<Double>>]) {
        solve(start: start, end: end, initialState: initialState, H: {_, _ in H }, stepSize: stepSize)
    }
    
    @inlinable
    public func solve(start: Double = .zero, end: Double, initialState: Matrix<Complex<Double>>, H: (Double, Matrix<Complex<Double>>) -> Matrix<Complex<Double>>, stepSize: Double = 0.01) -> (tSpace: [Double], densityMatrix: [Matrix<Complex<Double>>]) {
        precondition(initialState.rows == self.dimension)
        precondition(initialState.columns == self.dimension)
        var _initialState: Vector<Complex<Double>> = .zero(self.B.rows)
        for (i, value) in initialState.extractColumns().flatMap({ $0 }).enumerated() {
            _initialState[i] = value
        }
        let identity: Matrix<Complex<Double>> = .identity(rows: self.dimension)
        var _hamiltonianScratch: Matrix<Complex<Double>> = identity.kronecker(identity)
        var _currentState: Matrix<Complex<Double>> = .zeros(rows: dimension, columns: dimension)
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(B.rows), count: 4)
        return withoutActuallyEscaping(H) { H in
            var solver = RK4Solver(initialState: _initialState, t0: start, dt: stepSize) { t, currentState in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                for i in 0..<self.dimension {
                    for j in 0..<self.dimension {
                        _currentState[i, j] = currentState[i + self.dimension * j]
                    }
                }
                let H = H(t, _currentState)
                H.kronecker(identity, into: &_hamiltonianScratch)
                //TODO: Get rid of this allocation (H.transpose)
                identity.kronecker(H.transpose, multiplied: -.one, addingInto: &_hamiltonianScratch)
                
                result.components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        while index < resultBuffer.count {
                            _hamiltonianScratch.dot(currentStatePointer, multiplied: -.i, into: resultPointer)
                            resultPointer += dimension &* dimension
                            currentStatePointer += dimension &* dimension
                            index &+= dimension &* dimension
                        }
                    }
                }
                //B.dot(currentState, addingInto: &result)
                return result
            }
            
            var tSpace: [Double] = []
            var densityMatrix: [Matrix<Complex<Double>>] = []
            while solver.t <= end {
                let (t, state) = solver.step()
                tSpace.append(t)
                var rho: Matrix<Complex<Double>> = .zeros(rows: dimension, columns: dimension)
                for i in 0..<self.dimension {
                    for j in 0..<self.dimension {
                        rho[i, j] = state[i + j * self.dimension]
                    }
                }
                densityMatrix.append(rho)
            }
            return (tSpace, densityMatrix)
        }
    }
    
    @inlinable
    internal static func _constructBMatrix(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], lindbladians: [(Double, Matrix<Complex<Double>>)], truncationCondition: ([Int]) -> Bool) -> CSRMatrix<Complex<Double>> {
        let D = dimension * dimension
        let identity: Matrix<Complex<Double>> = .identity(rows: dimension)
        let kVectorPairs = _generateKVectorPairs(components: G.count, truncationCondition: truncationCondition)
        let positiveNeighbours = _generatePositiveNeighbours(kVectorPairs: kVectorPairs)
        let negativeNeighbours = _generateNegativeNeighbours(kVectorPairs: kVectorPairs)
        var result: LILMatrix<Complex<Double>> = .init(rows: D * kVectorPairs.count, columns: D * kVectorPairs.count)
        
        // Operators
        // L for vectorized operations (operates from left)
        let LVec = identity.kronecker(L)
        // LDagger for vectorized operations (operates from right)
        let LDaggerVec = L.conjugateTranspose.transpose.kronecker(identity)
        // [L, _] commutator
        let LCommutator = L.kronecker(identity) - identity.kronecker(L.transpose)
        // [L^\dagger, _] commutator
        let LDaggerCommutator = LCommutator.conjugateTranspose
        
        for (kVectorIndex, kVectorPair) in kVectorPairs.enumerated() {
            let row = kVectorIndex
            var dampingFactor: Complex<Double> = .zero
            for i in 0..<kVectorPair.0.count {
                dampingFactor += Double(kVectorPair.left[i]) * W[i]
                dampingFactor += Double(kVectorPair.right[i]) * W[i].conjugate
            }
            if dampingFactor != .zero {
                for i in 0..<D {
                    result[row * D + i, row * D + i] -= dampingFactor
                }
            }
            let positiveNeighbourIndices = positiveNeighbours[kVectorIndex]
            for (n, positiveNeighbourIndex) in positiveNeighbourIndices.left {
                let column = positiveNeighbourIndex
                let np1G = Double(kVectorPair.left[n] + 1).squareRoot() * G[n] / G[n].length.squareRoot()
                for i in 0..<D {
                    for j in 0..<D {
                        if LDaggerCommutator[i, j] != .zero {
                            result[row * D + i, column * D + j] += LDaggerCommutator[i, j] * np1G
                        }
                    }
                }
            }
            for (m, positiveNeighbourIndex) in positiveNeighbourIndices.right {
                let column = positiveNeighbourIndex
                let mp1G = Double(kVectorPair.right[m] + 1).squareRoot() * G[m].conjugate / G[m].length.squareRoot()
                for i in 0..<D {
                    for j in 0..<D {
                        if LCommutator[i, j] != .zero {
                            result[row * D + i, column * D + j] += LCommutator[i, j] * mp1G
                        }
                    }
                }
            }
            
            let negativeNeighbourIndices = negativeNeighbours[kVectorIndex]
            for (n, negativeNeighbourIndex) in negativeNeighbourIndices.left {
                let column = negativeNeighbourIndex
                let nG = Double(kVectorPair.left[n]).squareRoot() * G[n] / G[n].length.squareRoot()
                //let nG = Double(kVectorPair.left[n]) * G[n]
                for i in 0..<D {
                    for j in 0..<D {
                        if LVec[i, j] != .zero {
                            result[row * D + i, column * D + j] += LVec[i, j] * nG
                        }
                    }
                }
            }
            
            for (m, negativeNeighbourIndex) in negativeNeighbourIndices.right {
                let column = negativeNeighbourIndex
                let mG = Double(kVectorPair.right[m]).squareRoot() * G[m].conjugate / G[m].length.squareRoot()
                //let mG = Double(kVectorPair.right[m]) * G[m].conjugate
                for i in 0..<D {
                    for j in 0..<D {
                        if LDaggerVec[i, j] != .zero {
                            result[row * D + i, column * D + j] += LDaggerVec[i, j] * mG
                        }
                    }
                }
            }
            
        }
        return CSRMatrix(from: result)
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

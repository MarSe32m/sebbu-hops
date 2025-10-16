//
//  HOPSMultiParticleHierarchy.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 13.10.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms

/// Hierarchy structure for HOPS calculations
public struct HOPSMultiParticleHierarchy: Sendable {
    /// Shift type for the hierarchy.
    public enum ShiftType {
        /// No shift will be applied to the hierarchy.
        case none
        
        /// Mean field optimized shift will be applied to the hierarchy.
        /// Use this case when mean field approximation might be close to being a good approximation.
        /// This might not yield any benefits for small systems and low excitations in the environment.
        case meanField
        
        /// Exact shift to anchor the Q-function of the auxiliary state oscillators to the origin.
        /// This will yield the most optimal shift, however at the cost of performance.
        /// You might want to check also whether the mean field shift gives good results and use that to increase performance.
        //case exact
    }
    
    // Array containing the dot products kW
    @usableFromInline
    internal let kWArray: [Complex<Double>]
    // Sparse matrix containing the couplings between different levels of the hierarchy
    @usableFromInline
    internal let B: CSRMatrix<Complex<Double>>
    // Sparse matrix containing the indices of the coupling to the upper levels in the hierarchy
    @usableFromInline
    internal let P: [CSRMatrix<Complex<Double>>]
    // Sparse matrix containing the indicies of the coupling to the lower levels in the hierarchy
    @usableFromInline
    internal let N: [CSRMatrix<Complex<Double>>]
    // G coefficients for the BCF
    @usableFromInline
    internal let G: [Complex<Double>]
    // W coefficients for the BCF
    @usableFromInline
    internal let W: [Complex<Double>]
    
    /// The coupling operators
    public let L: [Matrix<Complex<Double>>]
    
    /// The system dimension
    public let dimension: Int
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - depth: The depth of the hierarchy
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], G: [[[Complex<Double>]]], W: [[[Complex<Double>]]], depth: Int) {
        self.init(dimension: dimension, L: L, G: G, W: W) { kTuple in
            kTuple.reduce(0, +) <= depth
        }
    }
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - truncationCondition: Truncation condition for the hierarchy truncation
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], G: [[[Complex<Double>]]], W: [[[Complex<Double>]]], truncationCondition: ([Int]) -> Bool) {
        precondition(G.count == W.count, "The G and W arrays must be of same size.")
        for L in L { precondition(dimension == L.columns) }
        precondition(L.count == G.count)
        precondition(G.count == W.count)
        for i in G.indices {
            precondition(G[i].count == W[i].count)
            precondition(G[i].count == L.count)
            for j in G[i].indices {
                precondition(G[i][j].count == W[i][j].count)
            }
        }
        
        let flatG = G.flatMap { $0 }.flatMap { $0 }
        let flatW = W.flatMap { $0 }.flatMap { $0 }
        let kTuples = HOPSMultiParticleHierarchy._generateKTuples(components: flatG.count, truncationCondition: truncationCondition)
        let positiveNeighbourIndices = HOPSMultiParticleHierarchy._generatePositiveNeighbourIndices(kTuples: kTuples)
        let negativeNeighbourIndices = HOPSMultiParticleHierarchy._generateNegativeNeighbourIndices(kTuples: kTuples)
        let kTupleComponentIndexMap = HOPSMultiParticleHierarchy._generateKTupleComponentIndexMap(systems: L.count, G: G)
        
        self.kWArray = HOPSHierarchy._generatekWArray(kTuples: kTuples, W: flatW)
        
        self.B = HOPSMultiParticleHierarchy._generateRescaledBMatrix(L: L, kTuples: kTuples, positiveNeighbourIndices: positiveNeighbourIndices, negativeNeighbourIndices: negativeNeighbourIndices, G: G)
        self.P = HOPSHierarchy._generateRescaledPMatrix(dimension: L.columns, kTuples: kTuples, G: G, positiveNeighbourIndices: positiveNeighbourIndices)
        self.N = HOPSHierarchy._generateRescaledNMatrix(dimension: L.columns, kTuples: kTuples, G: G, negativeNeighbourIndices: negativeNeighbourIndices)
        self.G = flatG
        self.W = flatW
        self.L = L
        self.dimension = dimension
    }
    
    @inlinable
    internal static func _generateKTuples(components: Int, kMax: Int) -> [[Int]] {
        var kVectors: [[Int]] = []
        for sum in 0...kMax {
            let partitions = sum.partitions(maxTerms: components).map { partition in
                if partition.count == components { return partition }
                precondition(partition.count < components)
                return partition + [Int](repeating: 0, count: components - partition.count)
            }
            for partition in partitions {
                for permutation in partition.uniquePermutations() {
                    assert(permutation.reduce(0, +) <= kMax, "The sum of the components exceeded the maximum allowed value \(sum), \(permutation)")
                    #if DEBUG
                    if kVectors.contains(permutation) {
                        fatalError("Duplicate k-vector")
                    }
                    #endif
                    kVectors.append(permutation)
                }
            }
        }
        return kVectors
    }
    
    @inlinable
    internal static func _generateKTuples(components: Int, truncationCondition: ([Int]) -> Bool) -> [[Int]] {
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
        return kVectors
    }

    @inlinable
    internal static func _generatePositiveNeighbourIndices(kTuples: [[Int]]) -> [[(component: Int, neighbourIndex: Int)]] {
        var index: [[Int]:Int] = [:]
        index.reserveCapacity(kTuples.count)
        for (i, kTuple) in kTuples.enumerated() {
            index[kTuple] = i
        }
        var result: [[(Int, Int)]] = []
        result.reserveCapacity(kTuples.count)
        for kTuple in kTuples {
            var neighbours: [(Int, Int)] = []
            var neighbour = kTuple
            for i in 0..<kTuple.count {
                neighbour[i] += 1; defer { neighbour[i] -= 1}
                if let neighbourIndex = index[neighbour] {
                    neighbours.append((i, neighbourIndex))
                }
            }
            result.append(neighbours)
        }
        return result
    }

    @inlinable
    internal static func _generateNegativeNeighbourIndices(kTuples: [[Int]]) -> [[(component: Int, neighbourIndex: Int)]] {
        var index: [[Int]:Int] = [:]
        index.reserveCapacity(kTuples.count)
        for (i, kTuple) in kTuples.enumerated() {
            index[kTuple] = i
        }
        var result: [[(Int, Int)]] = []
        result.reserveCapacity(kTuples.count)
        for kTuple in kTuples {
            var neighbours: [(Int, Int)] = []
            var neighbour = kTuple
            for i in 0..<kTuple.count {
                neighbour[i] -= 1; defer { neighbour[i] += 1}
                if let neighbourIndex = index[neighbour] {
                    neighbours.append((i, neighbourIndex))
                }
            }
            result.append(neighbours)
        }
        return result
    }

    @inlinable
    internal static func _generatekWArray(kTuples: [[Int]], W: [Complex<Double>]) -> [Complex<Double>] {
        kTuples.betterMap { kTuple in
            var result: Complex<Double> = .zero
            for i in 0..<kTuple.count {
                result -= Double(kTuple[i]) * W[i]
            }
            return result
        }
    }

    @inlinable
    internal static func _generateKTupleComponentIndexMap(systems: Int, G: [[[Complex<Double>]]]) -> [(n: Int, m: Int, mu: Int)] {
        var result: [(Int, Int, Int)] = []
        for n in 0..<systems {
            for m in 0..<systems {
                for mu in G[n][m].indices {
                    result.append((n, m, mu))
                }
            }
        }
        return result
    }
    
    @inlinable
    internal static func _generateBMatrix(L: [Matrix<Complex<Double>>], kTuples: [[Int]], G: [Complex<Double>],
                                          positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]],
                                          negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]],
                                          kTupleCoordinateMap: [(n: Int, m: Int, mu: Int)]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: L[0].rows * kTuples.count, columns: L[0].columns * kTuples.count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                let (n, _, _) = kTupleCoordinateMap[kIndex]
                let kG = Double(kTuple[kIndex] + 1).squareRoot() * .sqrt(G[kIndex])
                let LnDagger = L[n].conjugateTranspose
                for i in 0..<LnDagger.rows {
                    for j in 0..<LnDagger.columns {
                        if LnDagger[i, j] != .zero {
                            lilMatrix[row * LnDagger.rows + i, column * LnDagger.columns + j] = -kG * LnDagger[i, j]
                        }
                    }
                }
            }
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                if kTuple[kIndex] == .zero || G[kIndex] == .zero { continue }
                let (_, m, _) = kTupleCoordinateMap[kIndex]
                let kG = Double(kTuple[kIndex]).squareRoot() * .sqrt(G[kIndex])
                let Lm = L[m]
                for i in 0..<Lm.rows {
                    for j in 0..<Lm.columns {
                        if Lm[i, j] != .zero {
                            lilMatrix[row * Lm.rows + i, column * Lm.columns + j] = kG * Lm[i, j]
                        }
                    }
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }

    @inlinable
    internal static func _generatePMatrix(systems: Int, dimension: Int, kTuples: [[Int]],
                                          G: [Complex<Double>],
                                          positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: dimension * positiveNeighbourIndices.count, columns: dimension * positiveNeighbourIndices.count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                for i in 0..<dimension {
                    lilMatrix[row * dimension + i, column * dimension + i] = Complex(Double(kTuple[kIndex] + 1).squareRoot()) * .sqrt(G[kIndex])
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }
    
    @inlinable
    internal static func _generateRescaledNMatrix(dimension: Int, kTuples: [[Int]], G: [Complex<Double>], negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: dimension * negativeNeighbourIndices.count, columns: dimension * negativeNeighbourIndices.count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                for i in 0..<dimension {
                    lilMatrix[row * dimension + i, column * dimension + i] = Complex(Double(kTuple[kIndex]).squareRoot()) * .sqrt(G[kIndex])
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }
    
    /// Maps a HOPS trajectory to the corresponding density matrix
    /// - Parameters:
    ///   - trajectory: The HOPS trajectory to map to density matrix
    ///   - normalized: Whether the trajectory should be normalized. Default value is false.
    /// - Returns: Array of density matrices
    @inlinable
    public func mapTrajectoryToDensityMatrix(_ trajectory: [Vector<Complex<Double>>], normalize: Bool = false) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
            if normalize { rho[rho.count - 1] /= state.normSquared }
        }
        return rho
    }
}

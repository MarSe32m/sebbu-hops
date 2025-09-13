//
//  HOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms

public struct HOPSHierarchy: Sendable {
    // Array containing the dot products kW
    @usableFromInline
    internal let kWArray: [Complex<Double>]
    // Sparse matrix containing the couplings between different levels of the hierarchy
    @usableFromInline
    internal let B: CSRMatrix<Complex<Double>>
    // Sparse matrix containing the indices of the coupling to the upper levels in the hierarchy
    @usableFromInline
    internal let P: CSRMatrix<Complex<Double>>
    // G coefficients for the BCF
    @usableFromInline
    internal let G: [Complex<Double>]
    // W coefficients for the BCF
    @usableFromInline
    internal let W: [Complex<Double>]
    
    /// The coupling operator
    public let L: Matrix<Complex<Double>>
    
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
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], depth: Int) {
        precondition(G.count == W.count, "The G and W arrays must be of same size.")
        precondition(dimension == L.columns)
        let kTuples = HOPSHierarchy._generateKTuples(components: G.count, kMax: depth)
        self.kWArray = HOPSHierarchy._generatekWArray(kTuples: kTuples, W: W)
        let positiveNeighbourIndices = HOPSHierarchy._generatePositiveNeighbourIndices(kTuples: kTuples)
        let negativeNeighbourIndicies = HOPSHierarchy._generateNegativeNeighbourIndices(kTuples: kTuples)
        self.B = HOPSHierarchy._generateBMatrix(L: L, kTuples: kTuples, positiveNeighbourIndices: positiveNeighbourIndices, negativeNeighbourIndices: negativeNeighbourIndicies, G: G)
        self.P = HOPSHierarchy._generatePMatrix(dimension: L.columns, positiveNeighbourIndices: positiveNeighbourIndices)
        self.G = G
        self.W = W
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
    internal static func _generatePositiveNeighbourIndices(kTuples: [[Int]]) -> [[Int]] {
        var index: [[Int]:Int] = [:]
        index.reserveCapacity(kTuples.count)
        for (i, kTuple) in kTuples.enumerated() {
            index[kTuple] = i
        }
        var result: [[Int]] = []
        result.reserveCapacity(kTuples.count)
        for kTuple in kTuples {
            var neighbours: [Int] = []
            var neighbour = kTuple
            for i in 0..<kTuple.count {
                neighbour[i] += 1; defer { neighbour[i] -= 1}
                if let neighbourIndex = index[neighbour] {
                    neighbours.append(neighbourIndex)
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
    internal static func _generateBMatrix(L: Matrix<Complex<Double>>, kTuples: [[Int]], positiveNeighbourIndices: [[Int]], negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]], G: [Complex<Double>]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: L.rows * kTuples.count, columns: L.columns * kTuples.count)
        let Ldagger = L.conjugateTranspose
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            let negativeNeighbours = negativeNeighbourIndices[index]
            for positiveNeighbourIndex in positiveNeighbours {
                let column = positiveNeighbourIndex
                for i in 0..<Ldagger.rows {
                    for j in 0..<Ldagger.columns {
                        if Ldagger[i, j] != .zero {
                            lilMatrix[row * L.rows + i, column * L.columns + j] = -Ldagger[i, j]
                        }
                    }
                }
            }
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                if kTuple[kIndex] == .zero || G[kIndex] == .zero { continue }
                let kG = Double(kTuple[kIndex]) * G[kIndex]
                for i in 0..<L.rows {
                    for j in 0..<L.columns {
                        if L[i, j] != .zero {
                            lilMatrix[row * L.rows + i, column * L.columns + j] = kG * L[i, j]
                        }
                    }
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }

    @inlinable
    internal static func _generatePMatrix(dimension: Int, positiveNeighbourIndices: [[Int]]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: dimension * positiveNeighbourIndices.count, columns: dimension * positiveNeighbourIndices.count)
        for (index, positiveNeighbours) in positiveNeighbourIndices.enumerated() {
            let row = index
            for positiveNeighbourIndex in positiveNeighbours {
                let column = positiveNeighbourIndex
                for i in 0..<dimension {
                    lilMatrix[row * dimension + i, column * dimension + i] = .one
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }
    
    /// Map a linear HOPS trajectory to density matrix
    /// - Parameter trajectory: The linear HOPS trajectory to map to density matrix
    /// - Returns: Array of density matrices
    @inlinable
    @inline(__always)
    public func mapLinearToDensityMatrix(_ trajectory: [Vector<Complex<Double>>]) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
        }
        return rho
    }
    
    /// Maps a non-linear HOPS trajectory to density matrix
    /// - Parameter trajectory: The non-linear HOPS trajectory to map to density matrix
    /// - Returns: Array of density matrices
    @inlinable
    @inline(__always)
    public func mapNonLinearToDensityMatrix(_ trajectory: [Vector<Complex<Double>>]) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate) / state.normSquared)
        }
        return rho
    }
}

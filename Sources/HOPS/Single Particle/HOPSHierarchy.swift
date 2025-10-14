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

/// Hierarchy structure for HOPS calculations
public struct HOPSHierarchy: Sendable {
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
        case exact
    }
    
    // Array containing the dot products kW
    @usableFromInline
    internal let kWArray: [Complex<Double>]
    // Sparse matrix containing the couplings between different levels of the hierarchy
    @usableFromInline
    internal let B: CSRMatrix<Complex<Double>>
    // Sparse matrix containing the indices of the coupling to the upper levels in the hierarchy
    @usableFromInline
    internal let P: CSRMatrix<Complex<Double>>
    // Sparse matrix containing the indicies of the coupling to the lower levels in the hierarchy
    @usableFromInline
    internal let N: CSRMatrix<Complex<Double>>
    // G coefficients for the BCF
    @usableFromInline
    internal let G: [Complex<Double>]
    // W coefficients for the BCF
    @usableFromInline
    internal let W: [Complex<Double>]
    
    // "Creation" operators for the hierarchy
    @usableFromInline
    internal let creationOperators: [CSRMatrix<Complex<Double>>]
    
    // "Annihilation" operators for the hierarchy
    @usableFromInline
    internal let annihilationOperators: [CSRMatrix<Complex<Double>>]
    
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
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>], depth: Int, truncationCondition: (([Int]) -> Bool)? = nil) {
        precondition(G.count == W.count, "The G and W arrays must be of same size.")
        precondition(dimension == L.columns)
        //let kTuples = HOPSHierarchy._generateKTuples(components: G.count, kMax: depth)
        let kTuples = HOPSHierarchy._generateKTuples(components: G.count) { kTuple in
            truncationCondition?(kTuple) ?? (kTuple.reduce(0, +) <= depth)
        }
        self.kWArray = HOPSHierarchy._generatekWArray(kTuples: kTuples, W: W)
        let positiveNeighbourIndices = HOPSHierarchy._generatePositiveNeighbourIndices(kTuples: kTuples)
        let negativeNeighbourIndices = HOPSHierarchy._generateNegativeNeighbourIndices(kTuples: kTuples)
        self.B = HOPSHierarchy._generateRescaledBMatrix(L: L, kTuples: kTuples, positiveNeighbourIndices: positiveNeighbourIndices, negativeNeighbourIndices: negativeNeighbourIndices, G: G)
        self.P = HOPSHierarchy._generateRescaledPMatrix(dimension: L.columns, kTuples: kTuples, G: G, positiveNeighbourIndices: positiveNeighbourIndices)
        self.N = HOPSHierarchy._generateRescaledNMatrix(dimension: L.columns, kTuples: kTuples, G: G, negativeNeighbourIndices: negativeNeighbourIndices)
        self.creationOperators = HOPSHierarchy._generateCreationOperators(dimension: L.columns, kTuples: kTuples, negativeNeighbourIndices: negativeNeighbourIndices)
        self.annihilationOperators = HOPSHierarchy._generateAnnihilationOperators(dimension: L.columns, kTuples: kTuples, positiveNeighbourIndices: positiveNeighbourIndices)
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
                for permutation in partition.uniquePermutations().filter(truncationCondition) {
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
    internal static func _generateBMatrix(L: Matrix<Complex<Double>>, kTuples: [[Int]], positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]], negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]], G: [Complex<Double>]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: L.rows * kTuples.count, columns: L.columns * kTuples.count)
        let Ldagger = L.conjugateTranspose
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (_, positiveNeighbourIndex) in positiveNeighbours {
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
    internal static func _generatePMatrix(dimension: Int, positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: dimension * positiveNeighbourIndices.count, columns: dimension * positiveNeighbourIndices.count)
        for (index, positiveNeighbours) in positiveNeighbourIndices.enumerated() {
            let row = index
            for (_ ,positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                for i in 0..<dimension {
                    lilMatrix[row * dimension + i, column * dimension + i] = .one
                }
            }
        }
        return CSRMatrix(from: lilMatrix)
    }
    
    @inlinable
    internal static func _generateRescaledBMatrix(L: Matrix<Complex<Double>>, kTuples: [[Int]], positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]], negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]], G: [Complex<Double>]) -> CSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: L.rows * kTuples.count, columns: L.columns * kTuples.count)
        let Ldagger = L.conjugateTranspose
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                let kG = Double(kTuple[kIndex] + 1).squareRoot() * .sqrt(G[kIndex])
                for i in 0..<Ldagger.rows {
                    for j in 0..<Ldagger.columns {
                        if Ldagger[i, j] != .zero {
                            lilMatrix[row * L.rows + i, column * L.columns + j] = -kG * Ldagger[i, j]
                        }
                    }
                }
            }
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                if kTuple[kIndex] == .zero || G[kIndex] == .zero { continue }
                let kG = Double(kTuple[kIndex]).squareRoot() * .sqrt(G[kIndex])
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
    internal static func _generateRescaledPMatrix(dimension: Int, kTuples: [[Int]], G: [Complex<Double>], positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> CSRMatrix<Complex<Double>> {
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
    
    @inlinable
    internal static func _generateCreationOperators(dimension: Int, kTuples: [[Int]], negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> [CSRMatrix<Complex<Double>>] {
        var lilMatrices: [LILMatrix<Complex<Double>>] = .init(repeating: LILMatrix<Complex<Double>>(rows: dimension * negativeNeighbourIndices.count, columns: dimension * negativeNeighbourIndices.count), count: kTuples[0].count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                for i in 0..<dimension {
                    lilMatrices[kIndex][row * dimension + i, column * dimension + i] = Complex(Double(kTuple[kIndex]).squareRoot())
                }
            }
        }
        return lilMatrices.map { CSRMatrix(from: $0) }
    }
    
    @inlinable
    internal static func _generateAnnihilationOperators(dimension: Int, kTuples: [[Int]], positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]]) -> [CSRMatrix<Complex<Double>>] {
        var lilMatrices: [LILMatrix<Complex<Double>>] = .init(repeating: LILMatrix<Complex<Double>>(rows: dimension * positiveNeighbourIndices.count, columns: dimension * positiveNeighbourIndices.count), count: kTuples[0].count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                for i in 0..<dimension {
                    lilMatrices[kIndex][row * dimension + i, column * dimension + i] = Complex(Double(kTuple[kIndex] + 1).squareRoot())
                }
            }
        }
        return lilMatrices.map { CSRMatrix(from: $0) }
    }
    
    
    /// Maps a HOPS trajectory to the corresponding density matrix
    /// - Parameters:
    ///   - trajectory: The HOPS trajectory to map to density matrix
    ///   - normalized: Whether the trajectory should be normalized. Default value is false.
    /// - Returns: Array of density matrices
    @inlinable
    public func mapTrajectoryToDensityMatrix(_ trajectory: [Vector<Complex<Double>>], normalized: Bool = false) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
            if normalized { rho[rho.count - 1] /= state.normSquared }
        }
        return rho
    }
}

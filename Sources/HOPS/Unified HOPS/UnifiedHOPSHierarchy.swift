//
//  UnifiedHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 14.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms
import BasicContainers

/// Hierarchy structure for HOPS calculations
public struct UnifiedHOPSHierarchy: ~Copyable, Sendable {
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
    internal let kWArray: UniqueArray<Complex<Double>>
    
    // Sparse matrix containing the couplings between different levels of the hierarchy
    @usableFromInline
    internal let B: UniqueCSRMatrix<Complex<Double>>
    
    // Sparse matrix containing the indices of the coupling to the upper levels in the hierarchy
    @usableFromInline
    internal let P: UniqueArray<UniqueCSRMatrix<Complex<Double>>>
    
    // Sparse matrix containing the indicies of the coupling to the lower levels in the hierarchy
    @usableFromInline
    internal let N: UniqueArray<UniqueCSRMatrix<Complex<Double>>>
    
    // Sparse matrix for noise shifts for the non-linear HOPS
    @usableFromInline
    internal let M: UniqueCSRMatrix<Complex<Double>>
    
    // Indices for each of the shifts for the non-linear HOPS
    @usableFromInline
    internal let shiftIndices: UniqueArray<Range<Int>>
    
    // G coefficients for the BCFs
    @usableFromInline
    internal let G: UniqueArray<Complex<Double>>
    
    // W coefficients for the BCFs
    @usableFromInline
    internal let W: UniqueArray<Complex<Double>>
    
    // k-tuples for the hierarchy
    @usableFromInline
    internal let kTuples: UniqueArray<UniqueArray<Int>>
    
    /// The coupling operators
    public let L: UniqueArray<UniqueMatrix<Complex<Double>>>
    
    /// The coupling adjoint operators
    public let LDagger: UniqueArray<UniqueMatrix<Complex<Double>>>
    
    /// The system dimension
    public let dimension: Int
    
    /// The total dimension of the HOPS state vector
    public var totalDimension: Int { B.columns }
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations with independent noise processes
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - depth: The depth of the hierarchy
    @inlinable
    public init(dimension: Int, L: Matrix<Complex<Double>>, bathCorrelationFunctions: BathCorrelationFunction, depth: Int) {
        self.init(dimension: dimension, L: [L], bathCorrelationFunctions: [bathCorrelationFunctions], depth: depth)
    }
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations with independent noise processes
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - depth: The depth of the hierarchy
    @inlinable
    public init(dimension: Int, L: Matrix<Complex<Double>>, bathCorrelationFunctions: BathCorrelationFunction, truncationCondition: ([Int]) -> Bool) {
        self.init(dimension: dimension, L: [L], bathCorrelationFunctions: [bathCorrelationFunctions], truncationCondition: truncationCondition)
    }
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations with independent noise processes
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - depth: The depth of the hierarchy
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], bathCorrelationFunctions: [BathCorrelationFunction], depth: Int) {
        self.init(dimension: dimension, L: L, bathCorrelationFunctions: bathCorrelationFunctions) { kTuple in
            kTuple.reduce(0, +) <= depth
        }
    }
    
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], bathCorrelationFunctions: [BathCorrelationFunction], truncationCondition: ([Int]) -> Bool) {
        let bcfDimension = bathCorrelationFunctions.count
        var _bcfs: [[BathCorrelationFunction]] = []
        for i in 0..<bcfDimension {
            _bcfs.append([])
            for j in 0..<bcfDimension {
                if i == j {
                    _bcfs[i].append(bathCorrelationFunctions[j])
                } else {
                    _bcfs[i].append(.zero)
                }
            }
        }
        self.init(dimension: dimension, L: L, bathCorrelationFunctions: _bcfs, truncationCondition: truncationCondition)
    }
    
    
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], bathCorrelationFunctions: Matrix<BathCorrelationFunction>, depth: Int) {
        let bcfs = bathCorrelationFunctions.extractRows()
        self.init(dimension: dimension, L: L, bathCorrelationFunctions: bcfs, depth: depth)
    }
    
    /// Constructs a new HOPSHierarchy for subsequent trajectory calculations
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: The environment coupling operator
    ///   - G: The G coefficients of the bath correlation function exponential series
    ///   - W: The W exponents of the bath correlation function exponential series
    ///   - depth: The depth of the hierarchy
    @inlinable
    public init(dimension: Int, L: [Matrix<Complex<Double>>], bathCorrelationFunctions: [[BathCorrelationFunction]], depth: Int) {
        self.init(dimension: dimension, L: L, bathCorrelationFunctions: bathCorrelationFunctions) { kTuple in
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
    public init(dimension: Int, L: [Matrix<Complex<Double>>], bathCorrelationFunctions: [[BathCorrelationFunction]], truncationCondition: ([Int]) -> Bool) {
        for i in 0..<bathCorrelationFunctions.count {
            precondition(bathCorrelationFunctions[i].count == bathCorrelationFunctions.count, "The bath correlation functions must form a square matrix / grid")
        }
        for L in L { precondition(dimension == L.columns, "Dimension needs to match with the coupling operators") }
        precondition(L.count == bathCorrelationFunctions.count, "The number of coupling operators must match the number of bath correlation functions.")
        for i in bathCorrelationFunctions.indices {
            let bcfRow = bathCorrelationFunctions[i]
            precondition(bcfRow.count == L.count, "The number of coupling operators must match the number of bath correlation functions.")
        }
        var _G: [[[Complex<Double>]]] = []
        var _W: [[[Complex<Double>]]] = []
        for (i, bcfRow) in bathCorrelationFunctions.enumerated() {
            _G.append([])
            _W.append([])
            for bcfColumn in bcfRow {
                _G[i].append(bcfColumn.G)
                _W[i].append(bcfColumn.W)
            }
        }
        
        let flatG = _G.flatMap { $0 }.flatMap { $0 }
        let flatW = _W.flatMap { $0 }.flatMap { $0 }
        
        let kTuples = UnifiedHOPSHierarchy._generateKTuples(components: flatG.count, truncationCondition: truncationCondition)
        let positiveNeighbourIndices = UnifiedHOPSHierarchy._generatePositiveNeighbourIndices(kTuples: kTuples)
        let negativeNeighbourIndices = UnifiedHOPSHierarchy._generateNegativeNeighbourIndices(kTuples: kTuples)
        let kTupleComponentIndexMap = UnifiedHOPSHierarchy._generateKTupleComponentIndexMap(systems: L.count, G: _G)
        
        self.kWArray = UnifiedHOPSHierarchy._generatekWArray(kTuples: kTuples, W: flatW)
        self.B = UnifiedHOPSHierarchy._generateBMatrix(L: L, kTuples: kTuples, G: flatG, positiveNeighbourIndices: positiveNeighbourIndices, negativeNeighbourIndices: negativeNeighbourIndices, kTupleComponentIndexMap: kTupleComponentIndexMap)
        self.P = UnifiedHOPSHierarchy._generatePMatrices(systems: L.count, dimension: dimension, kTuples: kTuples, G: flatG, positiveNeighbourIndices: positiveNeighbourIndices, kTupleComponentIndexMap: kTupleComponentIndexMap)
        self.N = UnifiedHOPSHierarchy._generateNMatrices(systems: L.count, dimension: dimension, kTuples: kTuples, G: flatG, negativeNeighbourIndices: negativeNeighbourIndices, kTupleComponentIndexMap: kTupleComponentIndexMap)
        self.M = UnifiedHOPSHierarchy._generateShiftMatrix(systems: L.count, G: flatG, kTupleComponentIndexMap: kTupleComponentIndexMap)
        self.shiftIndices = UnifiedHOPSHierarchy._generateShiftIndices(systems: L.count, kTupleComponentIndexMap: kTupleComponentIndexMap)
        
        self.G = .init(copying: flatG)
        self.W = .init(copying: flatW)
        self.L = .init(capacity: L.count) { span in
            for O in L { span.append(.init(copying: O)) }
        }
        self.LDagger = .init(capacity: L.count) { span in
            for O in L { span.append(.init(copying: O.conjugateTranspose)) }
        }
        self.dimension = dimension
        self.kTuples = .init(capacity: kTuples.count) { span in
            for kTuple in kTuples { span.append(.init(copying: kTuple)) }
        }
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
    internal static func _generatekWArray(kTuples: [[Int]], W: [Complex<Double>]) -> UniqueArray<Complex<Double>> {
        var kWArray: UniqueArray<Complex<Double>> = .init()
        kTuples.forEach { kTuple in
            var sum: Complex<Double> = .zero
            for i in 0..<kTuple.count {
                sum -= Double(kTuple[i]) * W[i]
            }
            kWArray.append(sum)
        }
        return kWArray
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
                                          kTupleComponentIndexMap: [(n: Int, m: Int, mu: Int)]) -> UniqueCSRMatrix<Complex<Double>> {
        var lilMatrix = LILMatrix<Complex<Double>>(rows: L[0].rows * kTuples.count, columns: L[0].columns * kTuples.count)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                let (n, _, _) = kTupleComponentIndexMap[kIndex]
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
                let (_, m, _) = kTupleComponentIndexMap[kIndex]
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
        return UniqueCSRMatrix(from: lilMatrix)
    }

    @inlinable
    internal static func _generatePMatrices(systems: Int, dimension: Int, kTuples: [[Int]], G: [Complex<Double>],
                                            positiveNeighbourIndices: [[(component: Int, neighbourIndex: Int)]],
                                            kTupleComponentIndexMap: [(n: Int, m: Int, mu: Int)]) -> UniqueArray<UniqueCSRMatrix<Complex<Double>>> {
        var lilMatrices: [LILMatrix<Complex<Double>>] = .init(repeating: LILMatrix<Complex<Double>>(rows: dimension * positiveNeighbourIndices.count, columns: dimension * positiveNeighbourIndices.count), count: systems)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let positiveNeighbours = positiveNeighbourIndices[index]
            for (kIndex, positiveNeighbourIndex) in positiveNeighbours {
                let column = positiveNeighbourIndex
                let (n, _, _) = kTupleComponentIndexMap[kIndex]
                let kG = Double(kTuple[kIndex] + 1).squareRoot() * .sqrt(G[kIndex])
                for i in 0..<dimension {
                    lilMatrices[n][row * dimension + i, column * dimension + i] = kG
                }
            }
        }
        return .init(capacity: lilMatrices.count) { span in
            for matrix in lilMatrices {
                span.append(.init(from: matrix))
            }
        }
    }
    
    @inlinable
    internal static func _generateNMatrices(systems: Int, dimension: Int, kTuples: [[Int]], G: [Complex<Double>],
                                            negativeNeighbourIndices: [[(component: Int, neighbourIndex: Int)]],
                                            kTupleComponentIndexMap: [(n: Int, m: Int, mu: Int)]) -> UniqueArray<UniqueCSRMatrix<Complex<Double>>> {
        var lilMatrices: [LILMatrix<Complex<Double>>] = .init(repeating: LILMatrix<Complex<Double>>(rows: dimension * negativeNeighbourIndices.count, columns: dimension * negativeNeighbourIndices.count), count: systems)
        for (index, kTuple) in kTuples.enumerated() {
            let row = index
            let negativeNeighbours = negativeNeighbourIndices[index]
            for (kIndex, negativeNeighbourIndex) in negativeNeighbours {
                let column = negativeNeighbourIndex
                let (_, m, _) = kTupleComponentIndexMap[kIndex]
                let kG = Double(kTuple[kIndex]).squareRoot() * .sqrt(G[kIndex])
                for i in 0..<dimension {
                    lilMatrices[m][row * dimension + i, column * dimension + i] = kG
                }
            }
        }
        return .init(capacity: lilMatrices.count) { span in
            for matrix in lilMatrices {
                span.append(.init(from: matrix))
            }
        }
    }
    
    @inlinable
    internal static func _generateShiftMatrix(systems: Int, G: [Complex<Double>], kTupleComponentIndexMap: [(n: Int, m: Int, mu: Int)]) -> UniqueCSRMatrix<Complex<Double>> {
        var result = LILMatrix<Complex<Double>>(rows: G.count, columns: systems)
        for i in G.indices {
            let (_, m, _) = kTupleComponentIndexMap[i]
            result[i, m] = G[i].conjugate
        }
        return UniqueCSRMatrix(from: result)
    }
    
    @inlinable
    internal static func _generateShiftIndices(systems: Int,
                                              kTupleComponentIndexMap: [(n: Int, m: Int, mu: Int)]) -> UniqueArray<Range<Int>> {
        var result: UniqueArray<Range<Int>> = .init()
        var start = 0
        var end = 0
        for n in 0..<systems {
            while end < kTupleComponentIndexMap.count && kTupleComponentIndexMap[end].n == n { end += 1 }
            result.append(start..<end)
            start = end
        }
        return result
    }
}

public extension UnifiedHOPSHierarchy {
    @inlinable
    func fockStateAmplitudes(for trajectory: Trajectory, mode: Int, fockState: Int) -> [Double]? {
        guard let totalTrajectory = trajectory.totalTrajectory else { return nil }
        if mode >= kTuples[0].count { return .init(repeating: 0.0, count: totalTrajectory.count) }
        var auxiliaryIndices: [Int] = []
        for i in kTuples.indices {
            if kTuples[i][mode] == fockState {
                auxiliaryIndices.append(i)
            }
        }
        var amplitudes: [Double] = []
        for state in totalTrajectory {
            let normSquared = state.normSquared
            state.components.withUnsafeBufferPointer { stateBuffer in
                var amplitude: Double = .zero
                //TODO: Optimize this...
                for i in auxiliaryIndices {
                    let components = stateBuffer.baseAddress!.advanced(by: i * dimension)
                    let auxiliary = UniqueVector(_unsafeComponents: .init(mutating: components), count: dimension)
                    amplitude += auxiliary.normSquared
                    let _ = auxiliary.consumeComponents()
                }
                amplitudes.append(amplitude / normSquared)
            }
        }
        return amplitudes
    }
    
    struct Trajectory: Sendable {
        public let tSpace: [Double]
        public let systemTrajectory: [Vector<Complex<Double>>]
        public let totalTrajectory: [Vector<Complex<Double>>]?
        
        @inlinable
        @inline(always)
        init(tSpace: [Double], systemTrajectory: [Vector<Complex<Double>>], totalTrajectory: [Vector<Complex<Double>>]?) {
            self.tSpace = tSpace
            self.systemTrajectory = systemTrajectory
            self.totalTrajectory = totalTrajectory
        }
        
        /// Maps a HOPS trajectory to the corresponding density matrix
        /// - Parameters:
        ///   - trajectory: The HOPS trajectory to map to density matrix
        ///   - normalized: Whether the trajectory should be normalized.
        /// - Returns: Array of density matrices
        @inlinable
        @inline(always)
        public func densityMatrix(normalized: Bool) -> [Matrix<Complex<Double>>] {
            var rho: [Matrix<Complex<Double>>] = []
            rho.reserveCapacity(systemTrajectory.count)
            for state in systemTrajectory {
                rho.append(state.outer(state.conjugate))
                if normalized { rho[rho.count - 1] /= state.normSquared }
            }
            return rho
        }
        
        /// Computes the expectation value for the given operator
        /// - Parameters:
        ///   - O: Operator for which to compute the expectation value
        ///   - normalized: Whether the expectation value should be taken over normalized states
        /// - Returns: The expectation values for the trajectory
        @inlinable
        public func expectationValue(for O: Matrix<Complex<Double>>, normalized: Bool) -> [Complex<Double>] {
            var expectationValue: [Complex<Double>] = []
            expectationValue.reserveCapacity(systemTrajectory.count)
            for state in systemTrajectory {
                var _expectationValue = state.inner(state, metric: O)
                if normalized {
                    _expectationValue /= state.normSquared
                }
                expectationValue.append(_expectationValue)
            }
            return expectationValue
        }
    }
    
    struct CorrelationFunctionTrajectory: Sendable {
        public enum NormalizationSide: Sendable {
            case ket
            case bra
            case none
        }
        
        public let tSpace: [Double]
        public let systemKetTrajectory: [Vector<Complex<Double>>]
        public let systemBraTrajectory: [Vector<Complex<Double>>]
        public let totalKetTrajectory: [Vector<Complex<Double>>]?
        public let totalBraTrajectory: [Vector<Complex<Double>>]?
        public let normalizationSide: NormalizationSide
        
        @inlinable
        @inline(always)
        init(tSpace: [Double], systemKetTrajectory: [Vector<Complex<Double>>], systemBraTrajectory: [Vector<Complex<Double>>], totalKetTrajectory: [Vector<Complex<Double>>]?, totalBraTrajectory: [Vector<Complex<Double>>]?, normalizationSide: NormalizationSide) {
            self.tSpace = tSpace
            self.systemKetTrajectory = systemKetTrajectory
            self.systemBraTrajectory = systemBraTrajectory
            self.totalKetTrajectory = totalKetTrajectory
            self.totalBraTrajectory = totalBraTrajectory
            self.normalizationSide = .none
        }
        
        /// Maps the trajectory to the corresponding density matrix
        /// - Parameters:
        ///   - normalized: Whether the trajectory should be normalized.
        /// - Returns: Array of density matrices
        @inlinable
        @inline(always)
        public func densityMatrix(normalized: Bool) -> [Matrix<Complex<Double>>] {
            var rho: [Matrix<Complex<Double>>] = []
            rho.reserveCapacity(systemKetTrajectory.count)
            for (ket, bra) in zip(systemKetTrajectory, systemBraTrajectory) {
                rho.append(ket.outer(bra.conjugate))
                if normalized {
                    switch normalizationSide {
                    case .ket:
                        rho[rho.count - 1] /= ket.normSquared
                    case .bra:
                        rho[rho.count - 1] /= bra.normSquared
                    case .none:
                        break
                    }
                }
            }
            return rho
        }
    }
}

public extension UnifiedHOPSHierarchy {
    struct CustomOperator: @unchecked Sendable, ~Copyable {
        @usableFromInline
        let function: (Double, borrowing UniqueVector<Complex<Double>>, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let matrixStorage: UnsafeMutablePointer<Complex<Double>>?
        
        @inlinable
        deinit { matrixStorage?.deallocate() }
        
        @inlinable
        @inline(always)
        public init(_ matrix: Matrix<Complex<Double>>) {
            let matrixStorage = UniqueMatrix<Complex<Double>>(copying: matrix).consumeElements()
            let rows = matrix.rows
            let columns = matrix.columns
            self.function = { _, _, Heff in
                let O = UniqueMatrix<Complex<Double>>(_unsafeElements: matrixStorage, rows: rows, columns: columns)
                Heff.add(O)
                let _ = O.consumeElements()
            }
            self.matrixStorage = matrixStorage
        }
        
        @inlinable
        @inline(always)
        public init(_ function: @escaping @Sendable (Double, inout UniqueMatrix<Complex<Double>>) -> Void) {
            self.function = { t, _, Heff in function(t, &Heff) }
            self.matrixStorage = nil
        }
        
        @inlinable
        @inline(always)
        public init(_ function: @escaping @Sendable (Double, borrowing  UniqueVector<Complex<Double>>, inout UniqueMatrix<Complex<Double>>) -> Void) {
            self.function = function
            self.matrixStorage = nil
        }
        
        @inline(always)
        @inlinable
        public func callAsFunction(_ t: Double, _ state: borrowing UniqueVector<Complex<Double>>, addingTo Heff: inout UniqueMatrix<Complex<Double>>) {
            function(t, state, &Heff)
        }
    }
    
    struct JumpOperator<T: ComplexWhiteNoiseProcess>: ~Copyable {
        @usableFromInline
        internal let noise: T
        
        @usableFromInline
        internal let jumpOperator: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        internal let jumpOperatorDagger: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        internal let LDaggerL: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        internal let rate: @Sendable (Double) -> Double
        
        @inlinable
        @inline(always)
        public init(noise: T, rate: Double, jumpOperator: Matrix<Complex<Double>>) {
            self.init(noise: noise, rate: { _ in rate }, jumpOperator: jumpOperator)
        }
        
        @inlinable
        @inline(always)
        public init(noise: T, rate: @Sendable @escaping (Double) -> Double, jumpOperator: Matrix<Complex<Double>>) {
            self.noise = noise
            self.rate = rate
            self.jumpOperator = .init(copying: jumpOperator)
            self.LDaggerL = .init(copying: jumpOperator.conjugateTranspose.dot(jumpOperator))
            self.jumpOperatorDagger = .init(copying: jumpOperator.conjugateTranspose)
        }
        
        @inlinable
        @inline(always)
        internal func operate(on state: UnsafePointer<Complex<Double>>, into: UnsafeMutablePointer<Complex<Double>>) {
            jumpOperator.unsafeDot(state, into: into)
        }
    }
}

extension UnifiedHOPSHierarchy.JumpOperator: Sendable where T: Sendable {}

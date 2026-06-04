//
//  LinearHOPSState.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 28.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

extension UnifiedHOPSHierarchy {
    @usableFromInline
    internal struct LinearHOPSStateFunc<Noise: ComplexNoiseProcess>: ~Copyable, ~Escapable, ODERHSFunction {
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        var LDaggerExpectationValues: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let WConjugateVector: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var noiseShifts: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @usableFromInline
        let shiftType: ShiftType
        
        @_lifetime(copy noises, copy customOperators)
        @inlinable
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            shiftType: ShiftType,
            customOperators: consuming Span<CustomOperator>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.LDaggerExpectationValues = .zero(hierarchy.pointee.LDagger.count)
            self.WConjugateVector = .init(count: hierarchy.pointee.W.count) { buffer in
                for i in hierarchy.pointee.W.indices {
                    buffer[i] = -hierarchy.pointee.W[i].conjugate
                }
            }
            self.noiseShifts = .zero(noises.count)
            self.shiftType = shiftType
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y state: borrowing LinearHOPSState, dy result: inout LinearHOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            let LDaggerSpan = hierarchyPointer.pointee.LDagger.span
            
            let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            var LDaggerExpectations: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.LDaggerExpectationValues.components, count: LDaggerSpan.count)
            var noiseShifts: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.noiseShifts.components, count: noises.count)
            
            if shiftType == .meanField {
                // Noise / hierarchy shifting
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[unchecked: i] = systemState.inner(metric: LDaggerSpan[i], systemState) / normSquared
                }
                hierarchyPointer.pointee.M.dot(LDaggerExpectations.components, into: result.shiftVector.components)
                for i in 0..<result.shiftVector.count {
                    result.shiftVector[unchecked: i] = Relaxed.multiplyAdd(WConjugateVector[unchecked: i], state.shiftVector[unchecked: i], result.shiftVector[unchecked: i])
                }
                for i in hierarchyPointer.pointee.shiftIndices.indices {
                    noiseShifts[unchecked: i] = .zero
                    for j in hierarchyPointer.pointee.shiftIndices[i] {
                        noiseShifts[unchecked: i] = Relaxed.sum(noiseShifts[unchecked: i], state.shiftVector[unchecked: j])
                    }
                }
            }
            
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate)
                if shiftType == .meanField {
                    Heff.add(LDaggerSpan[unchecked: i], multiplied: -noiseShifts[unchecked: i].conjugate)
                }
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            if shiftType == .meanField {
                for i in hierarchyPointer.pointee.N.indices {
                    hierarchyPointer.pointee.N[i].dot(state.totalStateVector.components, multiplied: -LDaggerExpectations[i].conjugate, addingInto: result.totalStateVector.components)
                }
            }
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
            let _ = LDaggerExpectations.consumeComponents()
            let _ = noiseShifts.consumeComponents()
        }
        
        @inlinable
        public func zero() -> LinearHOPSState {
            return .init(dimension: totalDimension, shiftDimension: hierarchyPointer.pointee.G.count)
        }
    }
    
    @usableFromInline
    internal struct LinearHOPSState: ~Copyable, AdaptiveStepODESolverState {
        @inlinable
        public var norm: Double { totalStateVector.norm }
        
        public var totalStateVector: UniqueVector<Complex<Double>>
        public var shiftVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int, shiftDimension: Int) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(shiftDimension)
        }
            
        @inlinable
        public init(dimension: Int, initialShifts: borrowing Span<Complex<Double>>) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(initialShifts.count)
            for i in 0..<initialShifts.count {
                shiftVector[i] = initialShifts[i]
            }
        }

        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, shiftDimension: Int) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, initialShifts: borrowing Span<Complex<Double>>) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(initialShifts.count)
            for i in 0..<initialShifts.count {
                shiftVector[i] = initialShifts[i]
            }
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState) {
            totalStateVector.copyComponents(from: a.totalStateVector)
            shiftVector.copyComponents(from: a.shiftVector)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            totalStateVector.copyComponents(from: a.totalStateVector, multiplied: multiplied)
            shiftVector.copyComponents(from: a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public func distance(to: borrowing UnifiedHOPSHierarchy.LinearHOPSState) -> Double {
            let totalDistance = totalStateVector.squaredEuclideanDistance(to: to.totalStateVector) + shiftVector.squaredEuclideanDistance(to: to.shiftVector)
            return totalDistance.squareRoot()
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
            shiftVector.add(a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.LinearHOPSState, adding direction: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
            shiftVector.copyComponents(from: base.shiftVector, adding: direction.shiftVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
}

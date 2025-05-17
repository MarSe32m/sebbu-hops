//
//  HOPSOperator.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import NumericsExtensions

public struct HOPSOperator<T: AlgebraicField> {
    @usableFromInline
    @frozen
    internal enum Representation {
        case dense(Matrix<T>)
        case sparse(CSRMatrix<T>)
    }
    
    @usableFromInline
    internal var representation: Representation
    
    @inlinable
    public init(_ matrix: Matrix<T>) {
        self.representation = .dense(matrix)
    }
    
    @inlinable
    public init(_ matrix: CSRMatrix<T>) {
        self.representation = .sparse(matrix)
    }
    
    @inlinable
    @inline(__always)
    public func operate(on vector: Vector<T>, multiplied: T = 1, result: inout Vector<T>) {
        switch representation {
        case .dense(let matrix):
            matrix.dot(vector, multiplied: multiplied, into: &result)
        case .sparse(let matrix):
            matrix.dot(vector, multiplied: multiplied, into: &result)
        }
    }
    
    @inlinable
    @inline(__always)
    public func operate(on vector: Vector<T>, multiplied: T = 1, addingInto: inout Vector<T>) {
        switch representation {
        case .dense(let matrix):
            matrix.dot(vector, multiplied: multiplied, addingInto: &addingInto)
        case .sparse(let matrix):
            matrix.dot(vector, multiplied: multiplied, addingInto: &addingInto)
        }
    }
    
    @inlinable
    @inline(__always)
    public func operate(on vector: Vector<T>) -> Vector<T> {
        var result: Vector<T> = .zero(vector.count)
        operate(on: vector, addingInto: &result)
        return result
    }
}

extension HOPSOperator.Representation: Sendable where T: Sendable {}
extension HOPSOperator: Sendable where T: Sendable {}

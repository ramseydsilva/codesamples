'use strict';

/**
 * @file Simple package that extends Javascript's Object with two new methods.
 * Also see {@link https://github.com/ramseydsilva/oprop github}
 * and {@link https://www.npmjs.org/package/oprop npm}.
 * @author Ramsey D'silva <ramseydsilva@gmail.com>
 * @version 0.0.3
 */

/**
 * Creates and exposes Object's getProperty method
 * @method getProperty
 * @augments Object
 * @param {String} str Dot representation of the object's property
 * @returns {Object} Returns the value of object's nested property, else undefined
 */
Object.defineProperty(Object.prototype, 'getProperty', {
    set: function() {},
    get: function() {
        return function(str) {
            return str.split('.').reduce(function(o, x) {return (typeof o != 'undefined' && !!o[x] ? o[x] : undefined)}, this);
        }
    }
});

/**
 * Creates and exposes Object's hasProperty method
 * @method hasProperty
 * @augments Object
 * @param {String} str Dot representation of the object's property
 * @returns {Boolean} Wether this object contains the ennumerable property
 */
Object.defineProperty(Object.prototype, 'hasProperty', {
    set: function() {},
    get: function() {
        return function(str) {
            return !!this.getProperty(str);
        }
    }
});
'use strict';

var Post = require('../models/post'),
    _ = require('underscore'),
    gm = require('gm'),
    path = require('path'),
    fs = require('fs'),
    async = require('async'),
    ss = require('socket.io-stream'),
    postHelpers = require('../helpers/post'),
    albumHelpers = require('../helpers/album'),
    Album = require('../models/album'),
    postSocket = {};

/** This socket handles the post update page. It allows all users
 * logged in or anonymous to create city, restaurant, category, items
 * and ensure proper linkage. Successively, if user has logged in and is the owner
 * of the post in question, it allows an edit. Succisevely, it sends updated
 * post information to all post client listeners.
 */
postSocket.update = function(socket, callback) {
    socket.on('post-update', function(data) {
        Post.findOne({ _id: data.id }, function(err, post) {
            async.parallel([
                function(next) {
                    postHelpers.getOrCreateCityRestaurantCategoryItem(data.city, data.restaurant, data.category, data.item, next);
                }
            ], function(err, results) {
                if (post) {
                    console.log(socket.handshake.user);
                    if (post.userHasRights(socket.handshake.user)) {
                        var opts = {
                            title: data.title,
                            description: data.description,
                            _city: results[0].city.id,
                            _category: results[0].category.id,
                            _restaurant: results[0].restaurant.id,
                            _item: results[0].item.id
                        };
                        postHelpers.updatePost(post, opts, socket, 'post');

                    } else {
                        socket.emit('post', {error: 'Permission denied'});
                    }
                } else {
                    socket.emit('post', {error: 'Post not found'});

                    if (!!callback)
                        callback(err, post);

                }
            });
        });
    });
};

/** This socket will allow logged in user to upload
 * an image either individually or as an album. It
 * checks if the album flag has been checked to true.
 * If an album id is also specified, it adds the pictures
 * to the album. It also gathers city, restaurant, category
 * and item information and attaches it to the uploaded images
 * but not to the album (TODO)
 * If image has been uploaded to album, it sends new image information
 * to all client socket album listeners.
 */
postSocket.imageUpload = function(socket, io, callback) {
    ss(socket).on('image-upload', function(stream, data) {
        if (!!socket.handshake && !!!socket.handshake.user.id) {
            socket.emit('post', {error: 'Permission denied'});
        } else {
            var filename, elementId = data.elementId;

            if (!!data.link) {
                filename = path.basename(data.link).split('?')[0];
            } else {
                filename = path.basename(data.name.name);
            }
            async.waterfall([
                function(next) { // get file path
                    postHelpers.getFilePath(filename, 'original', socket.handshake.user, function(err, filepath) {
                        next(null, filepath);
                    });
                },
                function(filepath, next) { // get url
                    postHelpers.getImageUrl(filepath, function(url){
                        next(null, filepath, url);
                    });
                },
                function(filepath, url, next) { // start stream
                    if (!!data.link) {
                        var http = require('http'),
                            https = require('https'),
                            download = fs.createWriteStream(filepath);

                        var httpProtocol = !data.link.indexOf('https') ? https : http;
                        var httpStream = httpProtocol.get(data.link, function(res) {
                            res.pipe(download);
                            var totalSize = res.headers['content-length'];
                            var downloadedSize = 0;
                            res.on('data', function(buffer) {
                                downloadedSize += buffer.length;
                                socket.emit('downloadProgress', {'progress': downloadedSize/totalSize * 100 + '%' });
                            });
                        });
                        download.on('finish', function(err, data) { 
                            next(err, filepath, url, elementId);
                        });
                    } else {
                        var upload = fs.createWriteStream(filepath);
                        stream.pipe(upload);
                        stream.on('end', function(err, data) {
                            next(err, filepath, url, elementId);
                        });
                    }
                }
            ], function(err, filepath, url, elementId) {
                if (err) throw err;
                // Create city, restaurant, category first before saving post
                async.parallel({
                    album: function(next) {
                        if (!!data.album || !!data.create) {
                            albumHelpers.createOrGetAlbum(data.album, socket, io, next);
                        } else {
                            next(null, null);
                        }
                    },
                    cityRestaurantCategoryItem: function(next) {
                        postHelpers.getOrCreateCityRestaurantCategoryItem(data.city, data.restaurant, data.category, data.item, next);
                    },
                    thumb: function(next) {
                        postHelpers.getFilePath(path.basename(filepath), 'thumb/240x160', socket.handshake.user, function(err, thumbPath) {
                            gm(filepath).resize(240, 160, '!').gravity('Center').crop(240, 160, 0, 0).write(thumbPath, function(err) {
                                if (err) throw err;
                                postHelpers.getImageUrl(thumbPath, function(url){
                                    next(null, { path: thumbPath, url: url });
                                });
                            });
                        });
                    }
                }, function(err, results) {
                    var opts = {
                        user: {
                            _id: socket.handshake.user.id,
                            name: socket.handshake.user.profile.name
                        },
                        _user: socket.handshake.user.id,
                        pic: {
                            originalPath: filepath,
                            originalUrl: url,
                            thumbPath: results.thumb.path,
                            thumbUrl: results.thumb.url,
                        },
                        _city: results.cityRestaurantCategoryItem.city.id,
                        _restaurant: results.cityRestaurantCategoryItem.restaurant.id,
                        _category: results.cityRestaurantCategoryItem.category.id,
                        _item: results.cityRestaurantCategoryItem.item.id
                    };
                    async.waterfall([
                        function(next) {
                            albumHelpers.populatePostOptsWithAlbumValues(results.cityRestaurantCategoryItem, results.album, opts, next);
                        },
                        function(opts, next) {
                            Post(opts).save(next);
                        },
                        function(post, numberSaved, next) {
                            albumHelpers.assignPostToAlbum(post, results.album, socket.handshake.user, io, next);
                        }
                    ], function(err, results) {
                        postHelpers.socketNewPost(results.post, results.album, elementId, socket, callback);
                    });
                });
            });
        }
    });
};

/** This socket handles removal of post. It checks first
 * if the user has permission to the post
 */
postSocket.remove = function(io, socket, callback) {
    socket.on('post:remove', function(data) {
        if (!!!socket.handshake.user.id) {
            socket.emit('post', {error: 'Permission denied'});
        } else {
            Post.findOneAndRemove({_id: data.id, '_user': socket.handshake.user.id}, function(err, post) {
                if (!!post) {
                    post.remove();
                    var elements = {};
                    elements['#' + data.id] = 'Post deleted';

                    // if in new post page, remove element
                    io.sockets.in('post-new-' + post.id).emit('post-update', [{
                        action: 'remove',
                        elements: elements
                    }]);

                    // Redirect if on post delete page
                    io.sockets.in('post-delete-' + post.id).emit('post-update', [{
                        action: 'redirect',
                        url: socket.handshake.user.url
                    }]);
                } else {
                    socket.emit('post', {error: 'Post not found'});
                }

                if (!!callback)
                    callback(err, post);
            });
        }
    });
};

/** This socket handles user events on the post upload page, when they
 * delete any of the city, restaurant, category attributes by clicking
 * on the delete icon. It then updates the post realtime to all post listeneners
 */
postSocket.removeAttr = function(socket, attr, callback) {
    socket.on('post.' + attr + ':remove', function(data) {
        Post.findOne({_id: data.id, '_user': socket.handshake.user.id}, function(err, post) {
            post[attr] = null;
            post.save(function(err, doc) {
                var elements = {};
                elements['#' + data.id + ' .' + attr ] = "";
                socket.to('post-' + data.id).emit('post-update', [{
                    action: 'html',
                    elements: elements
                }]);

                if (!!callback)
                    callback(err, post);
            });
        });
    });
};

// Enables socket connection for post methods
postSocket.socket = function(io, socket, app) {
    postSocket.update(socket);
    postSocket.imageUpload(socket, io);
    postSocket.remove(io, socket);
    postSocket.removeAttr(socket, '_city');
    postSocket.removeAttr(socket, '_restaurant');
    postSocket.removeAttr(socket, '_category');
};

module.exports = postSocket;
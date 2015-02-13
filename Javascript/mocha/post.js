'use strict';

var request = require('supertest'),
    superagent = require('superagent'),
    _ = require('underscore'),
    jsdom = require('jsdom'),
    should = require('should'),
    ss = require('socket.io-stream'),
    async = require('async'),
    app = require('../../app'),
    Album = require('../../models/album'),
    User = require('../../models/User'),
    Post = require('../../models/post'),
    City = require('../../models/city'),
    Restaurant = require('../../models/restaurant'),
    Category = require('../../models/category'),
    Item = require('../../models/item'),
    userFixture = require('../fixtures/db/user'),
    util = require('../util'),
    postUtil = require('./util'),
    socketer = require('socketer'),
    fs = require('fs'),
    jquery = fs.readFileSync("node_modules/jquery/dist/jquery.min.js", "utf-8"),
    user;

var agent = superagent.agent(),
    user;

describe('Image uploading works', function() {
    var postLinkData = {
        elementId: '123',
        link: 'http://i.imgur.com/u40WWLD.jpg',
        city: 'Toronto',
        restaurant: 'Salad King',
        category: 'Thai',
        item: 'Noodles',
        album: false
    };

    var postImageData = {
        elementId: '456',
        name: {
            name: 'test/fixtures/poulet-roti.jpg',
            webkitRelativePath: ''
        },
        size: fs.statSync('test/fixtures/poulet-roti.jpg').size,
        city: 'Paris',
        restaurant: 'Petit France',
        category: 'French',
        item: 'Poulet roti',
        album: false
    };

    var postImageOnlyData = postImageData;
    postImageOnlyData['city'] = '';
    postImageData['restaurant'] = '';
    postImageData['category'] = '';
    postImageData['item'] = '';

    function uploadImage(socket, postData) {
        var stream = ss.createStream();
        ss(socket).emit('image-upload', stream, postData);
        if (!!postData.name) {
            var blobStream = fs.createReadStream(postData.name.name);
            blobStream.pipe(stream);
        }
    };

    before(function(done) {
        util.loadDb(
            function(next) {
                util.loadFixture(User, userFixture.user, next);
            }
        , function(err, results) {
            user = results[1];
            done(err);
        });
    });

    it('but anonymous user cannot post image, permission denied', function(done) {
        socketer.anonSocket(app, function(socket) {
            socket.once('connect', function() {
                uploadImage(socket, postLinkData);
            });
            socket.once('post', function(data) {
                socket.disconnect(); // Free the socket
               data.error.should.be.equal('Permission denied');
               done();
            });
        });
    });

    var newPostTests = [{
        name: 'New post can be created via link from internet, ensuring proper linkage',
        postData: postLinkData
    },{
        name: 'New post can be created via image upload, ensuring proper linkage',
        postData: postImageData
    }, {
        name: 'New post can be created without city, restaurant, category, item info',
        postData: postImageOnlyData
    }];

    _.each(newPostTests, function(testInfo) {
        describe(testInfo.name, function() {
            var post, postHtml, socket, postData = testInfo.postData;

            before(function(done) {
                util.loadDb(
                    function(next) {
                        util.loadFixture(User, userFixture.user, next);
                    }
                , function(err, results) {
                    user = results[1];

                    socketer.authSocket(app, {email: userFixture.user.email, password: userFixture.user.password}, '/login', function(authSocket) {
                        socket = authSocket;
                        socket.once('connect', function() {
                            uploadImage(socket, postData);
                        });
                        socket.once('post-update', function(data) {
                            socket.disconnect(); // Free the socket
                            Post.find({}, function(err, posts) {
                                post = posts[0];
                                postHtml = data[0].elements['#' + postData.elementId];
                                done();
                            });
                        });
                    });
                });
            });

            it('and should have an _id value', function() {
                post._id.should.be.ok;
            });

            it('and thumb was generated correctly', function() {
                post.pic.originalPath.should.be.ok;
                post.pic.originalUrl.should.be.ok;
                post.pic.thumbPath.should.be.ok;
                post.pic.thumbUrl.should.be.ok;
                fs.existsSync(post.pic.originalPath).should.be.true;
                fs.existsSync(post.pic.thumbPath).should.be.true;
            });

            it('and thumb is smaller than original image', function() {
                fs.statSync(post.pic.thumbPath).size.should.be.lessThan(fs.statSync(post.pic.originalPath).size);
            });

            it('and should have the user who created the post assigned to it', function() {
                post._user.should.be.eql(user._id);
            });

            it('and should have correct City, Restaurant, Category, Item linkage', function(done) {
                postUtil.ensureCityRestaurantCategoryItemLinkage(postData.city, postData.restaurant, postData.category, postData.item, function(err, results) {
                    done(err, results);
                });
            });

            it('and post html should contain city, restaurant, category names, and edit url', function() {
                postHtml.should.containEql(postData.city);
                postHtml.should.containEql(postData.restaurant);
                postHtml.should.containEql(postData.category);
            });

            it('and post html should contain 4 edit buttons and 4 delete buttons', function() {
                postHtml.match(/fa-times/g).length.should.be.exactly(4);
                postHtml.match(/fa-pencil/g).length.should.be.exactly(4);
            });

            it('and post html should contain thumb url', function() {
                postHtml.should.containEql(post.pic.thumbUrl);
            });

            it('and post page should contain user name and link to user page, but no edit and delete url for anon user', function(done) {
                socketer.anonRequest(app, post.url, function(err, res) {
                    res.body.should.containEql(user.url);
                    res.body.should.containEql(user.profile.name);
                    res.body.should.not.containEql(post.editUrl);
                    res.body.should.not.containEql(post.deleteUrl);
                    done(err);
                });
            });

            it('and post page should contain edit and delete url for user', function(done) {
                socketer.authRequest(app, post.url, {email: user.email, password: user.profile.passwordString}, function(err, res) {
                    res.body.should.containEql(post.editUrl);
                    res.body.should.containEql(post.deleteUrl);
                    done(err);
                });
            });

            it('and user profile page should contain post edit, delete buttons for user', function(done) {
                socketer.authRequest(app, user.url, {email: user.email, password: user.profile.passwordString}, function(err, res) {
                    res.body.should.containEql(post.editUrl);
                    res.body.should.containEql(post.deleteUrl);
                    done(err);
                });
            });

            it('and user profile page should not contain post edit, delete buttons for anon user', function(done) {
                socketer.anonRequest(app, user.url, function(err, res) {
                    res.body.should.not.containEql(post.editUrl);
                    res.body.should.not.containEql(post.deleteUrl);
                    done(err);
                });
            });

            it('and is not part of an album', function(done) {
                Album.findOne({pics: post.id}, function(err, doc) {
                    (!!doc).should.not.be.ok;
                    done(err);
                });
            });

            it('if album is created after post is added, it assigns post to album', function(done) {
                socketer.authSocket(app, {email: userFixture.user.email, password: userFixture.user.password}, '/login', function(authSocket) {
                    socket = authSocket;
                    socket.once('connect', function() {
                        socket.emit('create-album', {
                            album: '',
                            create: true,
                            pics: post.id
                        });
                    });
                    socket.once('create-album', function(albumData) {
                        socket.disconnect(); // Free the socket
                        setTimeout(function() {
                            Album.findOne({pics: post.id}, function(err, doc) {
                                (!!doc).should.be.ok;
                                doc.remove(done(err));
                            });
                        }, 500);
                    });
                });
            });

            it('and after post deleted, clear out the images', function(done) {
                post.remove(function(err, post) {
                    fs.existsSync(post.originalPath).should.be.false;
                    fs.existsSync(post.thumbPath).should.be.false;
                    done(err);
                });
            });

            after(function(done) { post.remove(done); });
        });
    });

    after(util.after);

});
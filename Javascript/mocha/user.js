'use strict';

var request = require('supertest'),
    superagent = require('superagent'),
    jsdom = require('jsdom'),
    should = require('should'),
    app = require('../../app.js'),
    User = require('../../models/User'),
    userFixture = require('../fixtures/db/user.js'),
    util = require('../util'),
    jquery = require('fs').readFileSync("node_modules/jquery/dist/jquery.min.js", "utf-8"),
    user;

var agent = superagent.agent();

var loadFixture = function(next) {
    user = User(userFixture.user);
    user.save(function(err, data) {
        next(err, data);
    });
};

describe('GET /users', function() {
    before(function(done) {
        util.loadDb(loadFixture, function(err, results) {
            user = results[1];
            done(err);
        });
    });

    it('should return 200 OK', function(done) {
        request(app).get('/users').expect(200, done);
    });

    it('Users page should contain link to user page', function(done) {
        request(app).get('/users').end(function(err, res) {
            res.text.should.containEql(user.profile.name);
            jsdom.env({html: res.text, src: [jquery], done: function (errors, window) {
                var $ = window.$;
                $('a[href="' + user.url + '"]').length.should.be.exactly(1);
                done();
            }});
        });
    });

    it('Profile page should redirect to login if not signed in', function(done) {
        request(app).get('/profile').end(function(err, res) {
            res.headers.location.should.be.exactly('/login');
            done();
        });
    });

    it('Profile page should redirect to user page if signed in', function(done) {
        request(app).post('/login').send({email: userFixture.user.email, password: userFixture.user.password}).expect(302).end(function(err, res) {
            if (!err) {
                agent.saveCookies(res);
                var req = request(app).get('/profile');
                agent.attachCookies(req);
                req.end(function(err, res2) {
                    res2.headers.location.should.be.exactly(user.url);
                    done(err);
                });
            } else {
                done(err);
            }
        });
    });

    it('User link should work', function(done) {
        var req = request(app).get(user.url);
        agent.attachCookies(req);
        req.expect(200, done);
    });

    it('Home page should show the following links', function(done) {
        var req = request(app).get('/');
        agent.attachCookies(req);
        req.end(function(err, res) {
            res.text.should.containEql('Cities').and.containEql('Categories').and.containEql('Home').and.containEql('Profile').and.containEql('Upload Picture');
            done(err);
        });
    });

    after(util.after);

});